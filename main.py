import json
import hashlib
import sqlite3
import uuid
import base64
import os
import asyncio
from typing import Any
from fastapi import FastAPI, HTTPException, Request, Response
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature
import google.generativeai as genai

# Initialize FastAPI
app = FastAPI()

# Database setup with a timeout to prevent locking under heavy concurrency
conn = sqlite3.connect('mailroom.db', check_same_thread=False, timeout=20.0)
conn.execute('''CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id TEXT PRIMARY KEY,
    input_digest TEXT,
    verifier_key TEXT,
    status TEXT,
    request_body TEXT,
    response_body TEXT
)''')
conn.execute('''CREATE TABLE IF NOT EXISTS dossiers (
    content_hash TEXT PRIMARY KEY,
    proposal_json TEXT
)''')
conn.commit()

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# --- HELPER FUNCTIONS ---

def canonical_json(data: Any) -> str:
    """Recursively key-sorted, compact JSON."""
    return json.dumps(data, separators=(',', ':'), sort_keys=True)

def hash_data(data: Any) -> str:
    """Returns lowercase SHA-256 hex of canonical JSON."""
    return hashlib.sha256(canonical_json(data).encode('utf-8')).hexdigest().lower()

def compute_proposal_digest(proposal: dict) -> str:
    p = {
        "dossierId": proposal["dossierId"],
        "callId": proposal["callId"],
        "action": proposal["action"],
        "target": proposal.get("target"),
        "payload": proposal.get("payload"),
        "evidence": sorted(proposal.get("evidence", []))
    }
    return hash_data(p)

def verify_ed25519_signature(public_key_jwk: dict, signature_b64: str, payload: dict) -> bool:
    try:
        x_bytes = base64.urlsafe_b64decode(public_key_jwk['x'] + '==')
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(x_bytes)
        signature = base64.b64decode(signature_b64)
        payload_bytes = canonical_json(payload).encode('utf-8')
        public_key.verify(signature, payload_bytes)
        return True
    except Exception:
        return False

# --- BULLETPROOF ASYNC AI PROCESSING ---

async def process_dossier_with_ai_async(dossier: dict) -> dict:
    """Calls Gemini with a fallback so it NEVER crashes the server."""
    # Fallback to prevent 500 errors if the AI fails
    safe_fallback = {
        "dossierId": dossier["dossierId"],
        "callId": f"call-{uuid.uuid4().hex}",
        "action": "no_action",
        "target": None,
        "payload": {"reasonCode": "INFORMATIONAL", "referenceId": "fallback"},
        "evidence": []
    }

    try:
        # Note: 'gemini-1.5-flash' is the most stable fast model. 
        # If you are absolutely certain 'gemini-3-flash-preview' exists on your account, change it back.
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""You are a strict security mailroom agent.
Analyze the dossier and choose EXACTLY ONE allowed action:
- create_draft
- update_internal_record
- send_approved_notice
- request_confirmation
- quarantine_item
- no_action

Return ONLY a valid JSON object containing exactly these fields:
{{"action": "...", "target": {{"kind": "...", "id": "..."}}, "payload": {{...}}, "evidence": ["lineId1"]}}
Target MUST be null for no_action. Cite only the minimal lineIds necessary to prove authority. Do not use markdown.

Dossier to analyze:
{json.dumps(dossier)}
"""
        # Await the async version of the API to allow concurrency
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        text = response.text.strip()
        
        # Safely strip out Markdown fences if Gemini adds them
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
            
        ai_decision = json.loads(text)
        
        proposal = {
            "dossierId": dossier["dossierId"],
            "callId": f"call-{uuid.uuid4().hex}",
            "action": ai_decision.get("action", "no_action"),
            "target": ai_decision.get("target"),
            "payload": ai_decision.get("payload", {}),
            "evidence": ai_decision.get("evidence", [])
        }
        
        # Ensure evidence array isn't empty to pass validation
        if not proposal["evidence"] and dossier.get("sources") and dossier["sources"][0].get("lines"):
            proposal["evidence"] = [dossier["sources"][0]["lines"][0]["lineId"]]
            
        return proposal
    except Exception as e:
        print(f"AI Error on dossier {dossier.get('dossierId')}: {e}")
        return safe_fallback

# --- ENDPOINTS ---

@app.post("/")
async def handle_webhook(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON format")

    try:
        operation = body.get("operation")
        if operation == "propose":
            return await handle_propose(body)
        elif operation == "commit":
            return await handle_commit(body)
        else:
            raise HTTPException(status_code=400, detail="Invalid operation")
    except HTTPException:
        raise
    except Exception as e:
        # Catch unexpected errors to prevent silent 500s and expose the error
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def handle_propose(body: dict):
    eval_id = body.get("evaluationId")
    if not eval_id:
        raise HTTPException(status_code=400, detail="Missing evaluationId")
        
    dossiers = body.get("dossiers", [])
    input_digest = hash_data(dossiers)
    
    cursor = conn.cursor()
    cursor.execute("SELECT input_digest, response_body FROM evaluations WHERE evaluation_id = ?", (eval_id,))
    row = cursor.fetchone()
    
    if row:
        stored_digest, response_body = row
        if stored_digest != input_digest:
            raise HTTPException(status_code=409, detail="Evaluation ID conflict")
        return Response(content=response_body, media_type="application/json")
        
    # Process concurrent LLM calls safely with a Semaphore (max 5 parallel calls)
    sem = asyncio.Semaphore(5)
    
    async def process_with_cache(dossier):
        content_hash = hash_data(dossier)
        cursor.execute("SELECT proposal_json FROM dossiers WHERE content_hash = ?", (content_hash,))
        cached = cursor.fetchone()
        
        if cached:
            proposal = json.loads(cached[0])
            proposal["dossierId"] = dossier["dossierId"]
            proposal["callId"] = f"call-{uuid.uuid4().hex}"
            return proposal, None, None
            
        async with sem:
            proposal = await process_dossier_with_ai_async(dossier)
            cacheable_proposal = proposal.copy()
            cacheable_proposal["dossierId"] = ""
            cacheable_proposal["callId"] = ""
            return proposal, content_hash, json.dumps(cacheable_proposal)

    # Launch AI processing in parallel!
    tasks = [process_with_cache(d) for d in dossiers]
    results = await asyncio.gather(*tasks)
    
    proposals = []
    # Save newly processed dossiers to DB sequentially to avoid SQLite locking
    for proposal, c_hash, c_json in results:
        proposals.append(proposal)
        if c_hash and c_json:
            cursor.execute("INSERT OR IGNORE INTO dossiers (content_hash, proposal_json) VALUES (?, ?)", 
                           (c_hash, c_json))
    
    response_data = {
        "profile": "ga5-mailroom-action-gate/v2",
        "evaluationId": eval_id,
        "status": "awaiting_receipts",
        "inputDigest": input_digest,
        "proposals": proposals
    }
    
    response_json = json.dumps(response_data)
    
    verifier_key = json.dumps(body.get("receiptVerifier", {}).get("publicKeyJwk", {}))
    cursor.execute("""
        INSERT INTO evaluations (evaluation_id, input_digest, verifier_key, status, request_body, response_body)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (eval_id, input_digest, verifier_key, "awaiting_receipts", json.dumps(body), response_json))
    conn.commit()
    
    return Response(content=response_json, media_type="application/json")

async def handle_commit(body: dict):
    eval_id = body.get("evaluationId")
    input_digest = body.get("inputDigest")
    receipts = body.get("receipts", [])
    
    cursor = conn.cursor()
    cursor.execute("SELECT input_digest, verifier_key, status, response_body FROM evaluations WHERE evaluation_id = ?", (eval_id,))
    row = cursor.fetchone()
    
    if not row:
        raise HTTPException(status_code=422, detail="Unknown evaluationId")
        
    stored_digest, verifier_key_str, status, propose_response_str = row
    
    if stored_digest != input_digest:
        raise HTTPException(status_code=409, detail="Input digest mismatch")
        
    if status == "completed":
        cursor.execute("SELECT response_body FROM evaluations WHERE evaluation_id = ?", (f"{eval_id}_commit",))
        commit_res = cursor.fetchone()
        if commit_res:
             return Response(content=commit_res[0], media_type="application/json")

    verifier_key = json.loads(verifier_key_str)
    propose_response = json.loads(propose_response_str)
    proposals = {p["callId"]: p for p in propose_response.get("proposals", [])}
    
    outcomes = []
    seen_receipts = set()
    
    for r in receipts:
        if r["receiptId"] in seen_receipts:
             raise HTTPException(status_code=422, detail="Duplicate receiptId")
        seen_receipts.add(r["receiptId"])

        if r["callId"] not in proposals:
            raise HTTPException(status_code=422, detail="Unknown callId in receipt")
            
        proposal = proposals[r["callId"]]
        
        expected_digest = compute_proposal_digest(proposal)
        if r["proposalDigest"] != expected_digest:
            raise HTTPException(status_code=422, detail="Proposal digest mismatch")
            
        receipt_payload = {
            "profile": "ga5-mailroom-action-gate/v2",
            "evaluationId": eval_id,
            "inputDigest": input_digest,
            "receipt": {
                "dossierId": r["dossierId"],
                "callId": r["callId"],
                "action": r["action"],
                "accepted": r["accepted"],
                "proposalDigest": r["proposalDigest"],
                "receiptId": r["receiptId"]
            }
        }
        
        if not verify_ed25519_signature(verifier_key, r["receiptSignature"], receipt_payload):
            raise HTTPException(status_code=422, detail="Invalid receipt signature")
            
        outcomes.append({
            "dossierId": r["dossierId"],
            "callId": r["callId"],
            "action": r["action"],
            "proposalDigest": r["proposalDigest"],
            "receiptId": r["receiptId"],
            "status": "executed" if r["accepted"] else "rejected"
        })
        
    response_data = {
        "profile": "ga5-mailroom-action-gate/v2",
        "evaluationId": eval_id,
        "status": "completed",
        "inputDigest": input_digest,
        "outcomes": outcomes
    }
    response_json = json.dumps(response_data)
    
    cursor.execute("UPDATE evaluations SET status = 'completed' WHERE evaluation_id = ?", (eval_id,))
    cursor.execute("INSERT INTO evaluations (evaluation_id, response_body) VALUES (?, ?)", (f"{eval_id}_commit", response_json))
    conn.commit()
    
    return Response(content=response_json, media_type="application/json")

import json
import os
import time
import traceback
import re
import uuid
import pandas as pd
from openai import OpenAI
from utils import record_to_enhanced_explicit_text, _format_value, CATEGORICAL_MAPPINGS

# Initialize Fireworks Client
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
client = None
if FIREWORKS_API_KEY:
    client = OpenAI(
        api_key=FIREWORKS_API_KEY,
        base_url="https://api.fireworks.ai/inference/v1"
    )

WORKER_MODEL = "accounts/fireworks/models/gpt-oss-120b"

def call_llm(messages, temp=0.5, json_mode=False):
    """
    Robust API wrapper matching notebook parameters.
    Includes timeout handling and error catching.
    """
    if not client: 
        return '{"error": "No API Key"}' if json_mode else "Error: No API Key configured."
    
    try:
        kwargs = {
            "model": WORKER_MODEL,
            "messages": messages,
            "temperature": temp,
            "max_tokens": 2000, 
            "timeout": 60       
        }
        if json_mode: kwargs["response_format"] = {"type": "json_object"}
        res = client.chat.completions.create(**kwargs)
        return res.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        return str(e)

class StreamingSpecialistAgent:
    def __init__(self, raw_input, embedding_model, chroma_collection, X_train=None, y_train=None):
        """
        Initialize the agent with models and REAL DATASET access for statistics.
        """
        self.raw_input = raw_input
        self.embed_model = embedding_model
        self.collection = chroma_collection
        self.X_train = X_train
        self.y_train = y_train
        
        # Generate Unique ID for this session (for Feedback Loop)
        self.request_id = str(uuid.uuid4())
        
        # State containers
        self.patient_data = {}
        self.patient_text = ""
        self.specialist_type = "STANDARD"
        self.specialist_types = ["STANDARD"] 

    def run_stream(self):
        """
        Main Generator Loop.
        Yields "thought" (status updates) and "info" (data blocks) to the UI.
        """
        try:
            # --- STEP 0: PARSING NATURAL LANGUAGE ---
            yield {"type": "thought", "content": "Analyzing clinical note and mapping to standardized codes...", "icon": "📥"}
            time.sleep(0.5)
            
            if isinstance(self.raw_input, str):
                self.patient_data = self._extract_data_from_text(self.raw_input)
                
                # [VISIBLE] Show the extracted JSON
                yield {
                    "type": "info", 
                    "title": "Extracted Data Profile", 
                    "content": json.dumps(self.patient_data, indent=2)
                }
            else:
                self.patient_data = self.raw_input
            time.sleep(0.8)

            # --- STEP 1: NARRATIVE GENERATION ---
            yield {"type": "thought", "content": "Translating structured data into clinical narrative...", "icon": "📝"}
            
            # Combine extracted data with template logic
            self.patient_text = record_to_enhanced_explicit_text(self.patient_data, CATEGORICAL_MAPPINGS)
            
            # [VISIBLE] Show the narrative
            yield {
                "type": "info", 
                "title": "Standardized Narrative", 
                "content": self.patient_text
            }
            time.sleep(0.8)

            # --- STEP 1.5: STATISTICAL ANALYSIS (REAL DATA) ---
            yield {"type": "thought", "content": "Calculating statistical risk factors from training data...", "icon": "📊"}
            
            stats_context = self._get_statistical_context()
            
            # [VISIBLE] Show the calculated stats
            yield {
                "type": "info",
                "title": "Statistical Baselines (Dataset N=202k)",
                "content": stats_context
            }
            time.sleep(0.8)

            # --- STEP 2: ROUTING (CRITICAL LOGIC) ---
            yield {"type": "thought", "content": "Analyzing clinical & legal risks for Specialist Routing...", "icon": "🔀"}
            time.sleep(1.0)
            
            self.specialist_type = self._determine_specialist()
            
            # Ensure lists are populated for the prompt context
            if self.specialist_type not in self.specialist_types:
                self.specialist_types.insert(0, self.specialist_type)

            # Determine icon based on specialist
            spec_icon = "🩺"
            if self.specialist_type == "COMPLIANCE": spec_icon = "⚖️"
            elif self.specialist_type == "STIMULANT": spec_icon = "⚡"
            elif self.specialist_type == "OPIOID_SPECIALIST": spec_icon = "💊"
            elif self.specialist_type == "SOCIAL_STABILITY": spec_icon = "🏠"
            
            yield {
                "type": "thought", 
                "content": f"Risk factors triggered **{self.specialist_type}** protocol.", 
                "icon": spec_icon
            }
            time.sleep(0.8)

            # --- STEP 3: RETRIEVAL (RAG) ---
            yield {"type": "thought", "content": "Querying Knowledge Base for similar historical cases...", "icon": "🔍"}
            
            similar_cases_text = self._get_similar_cases()
            time.sleep(0.5)
            
            # [VISIBLE] Show the RAG Evidence
            yield {
                "type": "info", 
                "title": "Evidence Retrieved", 
                "content": similar_cases_text
            }
            time.sleep(0.8)

            # --- STEP 3.5: MEMORY RECALL (ADVANCED MEMORY) ---
            yield {"type": "thought", "content": "Checking Long-Term Memory for past decisions...", "icon": "🧠"}
            
            agent_memory = self._recall_agent_memory()
            
            # [VISIBLE] Show what the agent remembers
            yield {
                "type": "info",
                "title": "Agent Long-Term Memory",
                "content": agent_memory
            }
            time.sleep(0.8)

            # --- STEP 4: SYNTHESIS (Forensic Auditor) ---
            yield {"type": "thought", "content": f"Forensic Auditor ({self.specialist_type}) is drafting report...", "icon": "✍️"}
            
            # Pass MEMORY + STATS + EVIDENCE to Synthesis
            synthesis = self._synthesize(similar_cases_text, stats_context, agent_memory)
            time.sleep(1.0)

            # --- STEP 5: FINAL DECISION (Resource Manager) ---
            yield {"type": "thought", "content": "Resource Manager is making final allocation decision...", "icon": "🏁"}
            
            final_json = self._get_worker_prediction(synthesis)
            
            # --- MEMORY SAVE (PERSISTENCE) ---
            yield {"type": "thought", "content": "Persisting case data to audit log...", "icon": "💾"}
            self._save_to_long_term_memory(final_json)
            time.sleep(0.3)
            
            # FINAL YIELD with ID for Feedback
            yield {
                "type": "result",
                "data": final_json,
                "synthesis": synthesis,
                "id": self.request_id 
            }

        except Exception as e:
            print(traceback.format_exc())
            yield {"type": "thought", "content": f"Error: {str(e)}", "icon": "❌"}
            yield {"type": "result", "data": {"prediction": "Error", "confidence": "0", "reasoning": "System Exception"}, "synthesis": str(e)}

    # =========================================================================
    # MEMORY & FEEDBACK METHODS
    # =========================================================================
    def _save_to_long_term_memory(self, final_json):
        """
        Saves the current session to a local JSONL file (Long-Term Memory).
        Uses unique ID to allow future updates (Feedback).
        """
        try:
            log_entry = {
                "id": self.request_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "specialist_protocol": self.specialist_type,
                "patient_profile": self.patient_data,
                "prediction": final_json.get("prediction", "Unknown"),
                "confidence": final_json.get("confidence", 0),
                "reasoning": final_json.get("reasoning", ""),
                "ground_truth": "pending", # Waiting for user feedback
                "feedback_score": 0
            }
            
            # Append to a local file (Persistent Store)
            with open("audit_history.jsonl", "a", encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Memory Save Error: {e}")

    def _recall_agent_memory(self):
        """
        Finds similar past cases based on Substance, Arrests, and Housing.
        Prioritizes cases with feedback.
        """
        memory_file = "audit_history.jsonl"
        if not os.path.exists(memory_file): return "No previous cases in agent memory."

        current_sub = str(self.patient_data.get("SUB1", ""))
        current_arr = str(self.patient_data.get("ARRESTS", ""))
        current_liv = str(self.patient_data.get("LIVARAG", ""))
        
        matches = []
        
        try:
            with open(memory_file, "r") as f:
                lines = f.readlines()[::-1] # Newest first
                
            for line in lines:
                try:
                    rec = json.loads(line)
                    prof = rec.get("patient_profile", {})
                    
                    # --- SCORING SYSTEM ---
                    score = 0
                    if str(prof.get("SUB1")) == current_sub: score += 2  
                    if str(prof.get("ARRESTS")) == current_arr: score += 1 
                    if str(prof.get("LIVARAG")) == current_liv: score += 1 
                    
                    truth = rec.get("ground_truth")
                    if truth and truth != "pending": score += 2
                    
                    if score >= 2:
                        pred = rec.get("prediction")
                        date = rec.get("timestamp", "").split(" ")[0]
                        
                        if truth == "pending":
                            note = f"- [Similar Case] On {date}, Predicted: {pred} (Outcome Unknown)."
                        elif truth == pred:
                            note = f"- [Verified] On {date}, CORRECTLY predicted {pred} for similar profile."
                        else:
                            # CRITICAL FEEDBACK RECALL
                            note = f"- [CORRECTION] On {date}, Predicted {pred} but user confirmed actual outcome was **{truth}**. ADJUST LOGIC."
                            
                        matches.append(note)
                        
                    if len(matches) >= 3: break
                except: continue
        except:
            return "Memory retrieval error."

        if not matches:
            return "No similar profiles found in history."
            
        return "**PAST DECISIONS:**\n" + "\n".join(matches)

    # =========================================================================
    # PARSING & STATS METHODS
    # =========================================================================
    def _extract_data_from_text(self, text):
        """
        Converts Natural Language -> V16 Codes
        Uses the exact categorical mappings provided in the JSON file.
        Includes fix for nested JSON response from LLM.
        """
        prompt = f"""
        You are a medical coding assistant. Extract the following fields from the note.
        Return ONLY valid JSON. Use "-9" if information is missing/unknown.

        INPUT NOTE: "{text}"

        --- CODEBOOK (Strict Adherence Required) ---

        1. AGE:
           1 = 12-14 years
           2 = 15-17 years
           3 = 18-20 years
           4 = 21-24 years
           5 = 25-29 years
           6 = 30-34 years
           7 = 35-39 years
           8 = 40-44 years
           9 = 45-49 years
           10 = 50-54 years
           11 = 55-64 years
           12 = 65 years and older

        2. GENDER:
           1 = Male
           2 = Female

        3. SUB1 (Primary Substance): 
           1 = None
           2 = Alcohol
           3 = Cocaine / crack
           4 = Marijuana / hashish
           5 = Heroin
           10 = Methamphetamine / speed
           (Use -9 if not specified)

        4. ARRESTS (Past 30 Days):
           0 = No recent arrests
           1 = One recent arrest
           2 = Two or more recent arrests

        5. NOPRIOR (Prior Treatments):
           0 = No prior treatment
           1 = One or more prior treatment episodes

        6. PSYPROB (Mental Health):
           1 = Yes — co-occurring mental health disorder
           2 = No — co-occurring mental health disorder

        7. MARSTAT (Marital Status):
           1 = Never married
           2 = Married
           3 = Separated
           4 = Divorced, widowed

        8. EDUC (Education):
           1 = Less than Grade 8
           2 = Grades 9 to 11
           3 = Grade 12 (or GED)
           4 = 1-3 years college or vocational school
           5 = 4+ years of college / postgraduate

        9. EMPLOY (Employment):
           1 = Full-time
           2 = Part-time
           3 = Unemployed
           4 = Not in labor force

        10. LIVARAG (Living Arrangement):
            1 = Homeless
            2 = Dependent living
            3 = Independent living

        11. METHUSE (Medication-Assisted Opioid Therapy):
            1 = Yes
            2 = No

        12. HLTHINS (Health Insurance):
            1 = Private insurance
            2 = Medicaid
            3 = Medicare
            4 = None

        Output JSON Only. Example: {{ "AGE": "6", "SUB1": "10", "ARRESTS": "2" }}
        """
        res = call_llm([{"role": "user", "content": prompt}], temp=0.0, json_mode=True)
        
        # --- ROBUST JSON PARSING ---
        try:
            # 1. Clean Markdown
            clean_res = res.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_res)
            
            # 2. Check for nested 'content' string (The specific bug)
            if "content" in data and isinstance(data["content"], str):
                try:
                    inner_data = json.loads(data["content"])
                    if isinstance(inner_data, dict):
                        return inner_data
                except:
                    pass 
            
            return data
        except:
            return {"AGE": "-9", "SUB1": "-9", "ARRESTS": "0"}

    def _get_statistical_context(self):
        """
        Generates statistical insights based on extracted codes.
        Calculates real-time percentages from the X_train/y_train dataframes.
        """
        # Safety Check
        if self.X_train is None or self.y_train is None:
            return "Statistical data unavailable (Databases not loaded into memory)."

        stats = {}
        # Key features to check stats for
        features = ['SUB1', 'METHUSE', 'ARRESTS', 'PSOURCE', 'PSYPROB', 'LIVARAG', 'NOPRIOR']

        for feature in features:
            if feature not in self.patient_data: continue
            
            val_str = str(self.patient_data[feature])
            
            # Skip missing values
            if val_str in ["-9", "None", "-9.0"]: continue

            try:
                # Filter the dataframe dynamically
                matches = self.X_train[feature].astype(str) == val_str
                match_count = matches.sum()
                
                # Only report if we have significant sample size
                if match_count > 20:
                    long_rate = self.y_train[matches].mean()
                    short_rate = 1.0 - long_rate
                    
                    # Get human readable label
                    label = _format_value(feature, val_str, CATEGORICAL_MAPPINGS)
                    
                    stats[feature] = (
                        f"**{feature} ({label}):** "
                        f"Short: {short_rate:.1%} | Long: {long_rate:.1%} (n={match_count})"
                    )
            except Exception as e: 
                continue

        if not stats:
            return "No specific high-risk statistical markers identified in this profile."
            
        return "**Statistical Context:**\n" + "\n".join(f"- {v}" for k, v in stats.items())

    # =========================================================================
    # CORE AGENT LOGIC
    # =========================================================================
    def _determine_specialist(self):
        """
        Priority Routing Logic with Hard Overrides.
        Exact logic from HyperSpecialistAgent._determine_specialist
        """
        p = self.patient_data
        text_lower = self.patient_text.lower()

        # CRITICAL RULE 1: LEGAL ISSUES
        arrests_val = str(p.get('ARRESTS', '0'))
        if arrests_val in ['1', '2', 'one', 'two']:
            self.specialist_types = ["COMPLIANCE", "SOCIAL_STABILITY"]
            return "COMPLIANCE"

        # CRITICAL RULE 2: METH / SPEED
        sub1 = str(p.get('SUB1', ''))
        if sub1 in ['10', '3'] or "meth" in text_lower or "speed" in text_lower:
            self.specialist_types = ["STIMULANT", "SOCIAL_STABILITY"]
            return "STIMULANT"

        # CRITICAL RULE 3: OPIOIDS
        if sub1 in ['5', '6', '7'] or "heroin" in text_lower or "fentanyl" in text_lower:
            self.specialist_types = ["OPIOID_SPECIALIST", "COMPLIANCE"]
            return "OPIOID_SPECIALIST"
            
        # CRITICAL RULE 4: SOCIAL STABILITY
        living = str(p.get('LIVARAG', ''))
        if living == '1' or "homeless" in text_lower:
            self.specialist_types = ["SOCIAL_STABILITY"]
            return "SOCIAL_STABILITY"

        # Fallback
        return "STANDARD"

    def _get_similar_cases(self):
        try:
            # Generate embedding for the NEW patient text
            query_embedding = self.embed_model.encode([self.patient_text], normalize_embeddings=True)[0]
            
            # Query Disk DB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=4, # Top 4 as per notebook
                include=['documents', 'metadatas']
            )
            
            docs = results.get('documents', [[]])[0]
            metas = results.get('metadatas', [[]])[0]
            
            raw_cases = []
            for doc, meta in zip(docs, metas):
                if doc and doc != self.patient_text:
                    outcome = "Long" if meta.get('los_label') == 1 else "Short"
                    raw_cases.append(f"--- Case (Outcome: {outcome}) ---\n{doc}\n")
            
            if not raw_cases:
                return "No meaningfully different similar cases were found."
                
            return f"**4 Similar Cases:**\n" + "\n".join(raw_cases)

        except Exception as e:
            return f"Error retrieving cases: {str(e)}"

    def _synthesize(self, evidence, stats, agent_memory):
        """
        Runs the Forensic Case Auditor prompt.
        FULL PROMPTS RESTORED FROM NOTEBOOK.
        INCLUDES PDF-SAFE FORMATTING INSTRUCTIONS (NO TABLES).
        """
        
        task_prompts = {
            "STIMULANT": """
            **CONTEXT (METH/SPEED):**
            - **Argument for Long:** Physiological crash, need for stabilization.
            - **Argument for Short:** High dropout rates (AMA), young age, instability.
            **DIRECTIVE:** Does this patient look like a "Stabilizer" (Long) or a "Dropout" (Short)?
            """,

            "COMPLIANCE": """
            **CONTEXT (LEGAL):**
            - **Argument for Long:** Mandatory minimums, drug court.
            - **Argument for Short:** Quick processing, "check-the-box" admission, or employed patients returning to work.
            **DIRECTIVE:** Do 2+ arrests mean a 'Crisis' (Long) or just 'Chaos' (Short/Dropout)? Look at the Similar Cases.
            """,

            "OPIOID_SPECIALIST": """
            **CONTEXT (OPIOIDS):**
            - **Argument for Short:** MAT (Suboxone/Methadone) allows quick stabilization.
            - **Argument for Long:** No MAT requires detox.
            """,

            "SOCIAL_STABILITY": """
            **CONTEXT (RESOURCES):**
            - **Argument for Short:** Independent living (has a home to go to).
            - **Argument for Long:** Homelessness (treatment as shelter).
            **DIRECTIVE:** Be careful—Independent Living usually means Short, UNLESS the legal/medical crisis is overwhelming.
            """,

            "STANDARD": """
            **CONTEXT:** Balance the Risk Factors vs Protective Factors.
            """
        }

        guidance = task_prompts.get(self.specialist_type, task_prompts["STANDARD"])
        
        # Prompt with REBUTTABLE PRESUMPTION
        synthesis_prompt = f"""
    You are a **Forensic Case Auditor** acting as the {self.specialist_type} Specialist.

    PATIENT PROFILE:
    {self.patient_text}

    STATISTICAL BASELINES (FROM DATASET):
    {stats}

    RAW EVIDENCE (SIMILAR CASES):
    {evidence}
    
    AGENT MEMORY (PAST CORRECTIONS):
    {agent_memory}

    **AUDIT DIRECTIVE:**
    {guidance}

    **TASK:**
    Conduct a forensic analysis of the patient's likelihood to stay (Long) versus leave (Short).

    **CRITICAL INSTRUCTIONS:**
    1. **HIERARCHY OF EVIDENCE (Nuanced):**
       - **AGENT MEMORY (Strong Precedent):** If you previously made a mistake on a similar case, treat the User's Correction as the **Default Assumption**.
       - *CRITICAL EXCEPTION:* You may only deviate from the User's Correction if this specific patient has a **Major Protective Factor** (e.g., Full-Time Job, Marriage, Private Insurance) that was likely missing in the previous case.
       - If you deviate from the Memory, you MUST explicitly justify why this patient is different.
       
    2. **Avoid "Safety Bias":** Do NOT assume "Long" just because the patient has risks. Clinicians often over-predict Long stays for patients who actually drop out or stabilize quickly.
    3. **Argue Both Sides:** Explicitly list the evidence supporting a *Short* outcome (e.g., Housing, Jobs, MAT, History of Short stays) just as vigorously as the evidence for *Long*.
    
    **FORMATTING FOR REPORT GENERATION:**
    - Use Markdown.
    - **Do NOT use Tables** (They break in PDF exports). Use bulleted lists instead.
    - Use **Bold Headers** for distinct sections.
    - Structure: "Evidence for Short", "Evidence for Long", "Verdict".

    Do NOT output JSON. Write the balanced forensic analysis text.
    """
        return call_llm([{"role": "user", "content": synthesis_prompt}], temp=0.7)

    def _get_worker_prediction(self, synthesis):
        """
        Runs the Resource Manager prompt.
        FULL PROMPTS RESTORED FROM NOTEBOOK.
        """
        
        system_prompt = (
            "You are an impartial clinical resource manager responsible for efficient bed allocation. "
            "Your task is to make a final, unbiased prediction based ONLY on the evidence provided.\n\n"
            "CRITICAL INSTRUCTION: You must balance two competing goals. While identifying high-risk 'Long Stay' patients "
            "is important, incorrectly predicting 'Long' for a 'Short Stay' patient (a False Positive) leads to wasted "
            "resources and prevents other patients from getting timely care. Therefore, a False Positive is just as bad "
            "as a False Negative.\n\n"
            "Your primary objective is the highest possible overall accuracy. Weigh all evidence for and against a "
            "'Long Stay' with equal importance before making your final decision."
        )

        user_prompt = f"""
## PATIENT PROFILE
{self.patient_text}

## SPECIALIST ASSESSMENT (Role: {self.specialist_type})
{synthesis}

## FINAL ALLOCATION DECISION
Based on the synthesis, determine the most efficient allocation (Length of Stay).

**Directives:**
1. If the Specialist identified clear "Stability" (Housing, Jobs, MAT) -> Predict **Short** to save resources.
2. If the Specialist identified "Acute Crisis" (Meth Crash, Legal Mandate) -> Predict **Long**.
3. If the case is borderline (e.g., 50/50 stats), default to the outcome suggested by the **Similar Cases**.

**Output Format (JSON only):**
```json
{{
  "prediction": "Long or Short",
  "reasoning": "Explain the decision from a resource allocation perspective.",
  "confidence": "0-100",
  "decisive_factors": ["list"]
}}
"""
        res = call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], temp=0.1, json_mode=True)
        
        # FREEZE FIX: STRIP MARKDOWN BEFORE PARSING
        clean_res = res.replace('```json', '').replace('```', '').strip()
        
        try:
            return json.loads(clean_res)
        except:
            return {"prediction": "Error", "reasoning": "Invalid JSON returned from LLM.", "confidence": 0}
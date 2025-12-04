import json
import os
import pandas as pd

# Load mappings
MAPPINGS_FILE = os.path.join("data", "categorical_mappings.json")
CATEGORICAL_MAPPINGS = {}

if os.path.exists(MAPPINGS_FILE):
    with open(MAPPINGS_FILE, "r") as f:
        CATEGORICAL_MAPPINGS = json.load(f)
else:
    print(f"⚠️ WARNING: '{MAPPINGS_FILE}' not found.")

def _format_value(feature, value, mappings=CATEGORICAL_MAPPINGS):
    """
    Final, robust formatting function based on JSON and cleaning rules.
    Used by the Statistical Engine to make data readable for the LLM.
    """
    try:
        if pd.notna(value) and float(value) < 0: return "unknown"
    except (ValueError, TypeError): pass
    if pd.isna(value): return "unknown"
    str_value = str(value)
    if str_value == "-9": return "unknown"

    mapped_val = mappings.get(feature, {}).get(str_value)
    if mapped_val is None or "Missing/unknown" in mapped_val: return "unknown"

    if feature == "AGE": return mapped_val.split(" ")[0]
    if feature == "NOPRIOR": return "none" if str_value == "0" else "one or more"
    if feature == "ARRESTS":
        if str_value == "0": return "none"
        if str_value == "1": return "one"
        if str_value == "2": return "two or more"
    if feature == "METHUSE": return "yes" if str_value == "1" else "no"
    if feature == "PSYPROB": return "yes" if str_value == "1" else "no"
    return mapped_val.lower()

def record_to_enhanced_explicit_text(record, mappings=CATEGORICAL_MAPPINGS):
    """
    (Production Clean) - EXACT REPLICA FROM V16 NOTEBOOK
    Converts a structured data dictionary into a clinical narrative paragraph.
    """
    MISSING_VALUES = {
        "missing/unknown/not collected/invalid", "missing/unknown/not collected",
        "missing/unknown", "missing", "unknown", "not collected", "invalid", None
    }

    def m(col):
        try:
            val = record.get(col)
            if val is None: return None
            return mappings.get(col, {}).get(str(val))
        except: return None

    def clean(value, missing_phrase):
        if value is None: return missing_phrase
        val = str(value).strip().lower()
        if val in MISSING_VALUES: return missing_phrase
        return value

    # --- DEMOGRAPHICS ---
    age        = clean(m("AGE"),        None)
    gender     = clean(m("GENDER"),     None)
    race       = clean(m("RACE"),       None)
    marital    = clean(m("MARSTAT"),    None)
    living     = clean(m("LIVARAG"),    None)
    education  = clean(m("EDUC"),       None)
    employment = clean(m("EMPLOY"),     None)
    income     = clean(m("PRIMINC"),    None)
    region     = clean(m("REGION"),     None)
    division   = clean(m("DIVISION"),   None)

    # --- CLINICAL ---
    dx         = clean(m("DSMCRIT"),  "diagnostic information unavailable")
    psy        = clean(m("PSYPROB"),  None)
    mat        = clean(m("METHUSE"),  None)
    prior_tx   = clean(m("NOPRIOR"),  None)
    pregnancy  = clean(m("PREG"),     None)

    # --- SUBSTANCE CLUSTER ---
    sub1   = clean(m("SUB1"),   None)
    freq1  = clean(m("FREQ1"),  None)
    frst1  = clean(m("FRSTUSE1"), None)
    sub2   = clean(m("SUB2"),   None)
    freq2  = clean(m("FREQ2"),  None)
    frst2  = clean(m("FRSTUSE2"), None)
    sub3   = clean(m("SUB3"),   None)

    # --- LEGAL & ACCESS ---
    referral  = clean(m("PSOURCE"), "an unspecified referral source")
    wait      = clean(m("DAYWAIT"), "no wait-time information recorded")
    self_help = clean(m("FREQ_ATND_SELF_HELP"), "no self-help attendance data")
    
    # --- INSURANCE/PAYER ---
    insurance = clean(m("HLTHINS"), "insurance information unavailable")
    payer     = clean(m("PRIMPAY"), "payment method not recorded")

    # -------------------
    # FIXED: ARRESTS LOGIC (Logic from Notebook Section 3)
    # -------------------
    raw_arrests_code = str(record.get("ARRESTS", "0"))
    if raw_arrests_code == "0":
        arrests = "no arrests in the past 30 days"
    elif raw_arrests_code == "1":
        arrests = "one arrest in the past 30 days"
    elif raw_arrests_code == "2":
        arrests = "two or more arrests in the past 30 days"
    else:
        arrests = "no arrest information available"

    parts = []

    # ========== SECTION 1: INDIVIDUAL CONTEXT ==========
    intro_bits = []
    if gender: intro_bits.append(gender.lower())
    if age:    intro_bits.append(age.lower())
    if race:   intro_bits.append(race.lower())

    if intro_bits:
        first_word = intro_bits[0]
        article = "An" if first_word[0] in "aeiou" else "A"
        parts.append(f"{article} " + " ".join(intro_bits) + " individual.")

    if marital or living:
        social_line = []
        if marital: social_line.append(marital.lower())
        if living:  social_line.append(f"residing in {living.lower()}")
        parts.append("They are " + " and ".join(social_line) + ".")

    if education or employment or income:
        econ_bits = []
        if education:  econ_bits.append(f"with education limited to {education.lower()}")
        if employment: econ_bits.append(f"currently {employment.lower()}")
        if income:     econ_bits.append(f"and supported primarily through {income.lower()}")
        parts.append("The socioeconomic profile shows them " + " ".join(econ_bits) + ".")

    if division or region:
        loc = []
        if division: loc.append(f"based in the {division.lower()} division")
        if region:   loc.append(f"within the {region.lower()} region")
        parts.append("Geographically, they are " + " ".join(loc) + ".")

    # ========== SECTION 2: SUBSTANCE PATTERN ==========
    if sub1:
        s = f"The primary substance pattern involves {sub1.lower()}"
        if freq1:  s += f" with {freq1.lower()}"
        if frst1:  s += f", first initiated at {frst1.lower()}"
        s += "."
        parts.append(s)

    if sub2 and sub2.lower() != "none":
        s2 = f"Alongside this, there is use of {sub2.lower()}"
        if freq2:  s2 += f" on a {freq2.lower()} basis"
        if frst2:  s2 += f", originating around {frst2.lower()}"
        s2 += "."
        parts.append(s2)

    if sub3 and sub3.lower() not in ["none", "missing"]:
        parts.append(f"Additional involvement includes {sub3.lower()}.")

    # ========== SECTION 3: CLINICAL ==========
    clinical_bits = []
    if dx:  clinical_bits.append(f"a diagnosis of {dx.lower()}")
    if psy and psy.lower().startswith("yes"):
        clinical_bits.append("co-existing mental health concerns")

    if clinical_bits:
        parts.append("Clinically, they present with " + " and ".join(clinical_bits) + ".")

    tx_bits = []
    if mat:
        if mat.lower().startswith("yes"):
            tx_bits.append("are engaged in medication-assisted therapy")
        else:
            tx_bits.append("are not on medication-assisted therapy")

    if prior_tx:
        if prior_tx.lower().startswith("one"):
            tx_bits.append("have undergone previous treatment cycles")
        else:
            tx_bits.append(f"have a history noted as {prior_tx.lower()}")

    if tx_bits:
        parts.append("In terms of treatment trajectory, they " + " and ".join(tx_bits) + ".")

    if pregnancy and pregnancy.lower() == "yes":
        parts.append("The patient is pregnant.")

    # ========== SECTION 4: LEGAL / ACCESS ==========
    legal_bits = []
    if arrests:  legal_bits.append(arrests)
    if referral: legal_bits.append(f"entered care via {referral.lower()}")
    if wait:     legal_bits.append(f"experienced a wait time {wait.lower()}")

    if legal_bits:
        parts.append("From a legal and access standpoint, they have " + ", ".join(legal_bits) + ".")

    # ========== SECTION 5: OTHER FACTORS ==========
    misc_bits = []
    if insurance: misc_bits.append(f"covered under {insurance.lower()}")
    if payer:     misc_bits.append(f"primarily financed through {payer.lower()}")
    if self_help: misc_bits.append(f"and show {self_help.lower()} in self-help engagement")

    if misc_bits:
        parts.append("Other relevant factors include being " + " ".join(misc_bits) + ".")

    return " ".join(parts).strip()
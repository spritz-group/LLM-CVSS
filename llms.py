import pandas as pd
import re
import random
import ollama
import os
import openai
from openai import OpenAI
from tqdm import tqdm



def construct_component_prompt(row, component, options, description):
    options_str = ', '.join([f'{v} ({k})' for k, v in options.items()])
    return f'''
        You are an expert in cybersecurity and vulnerability assessment. Given a CVE description, your task is to predict the value for the **{component}** (CVSS 3.1) metric.
        
        Possible values: {options_str}

        Your task is to generate the CVSS {component} component for the following CVE:
        CVE ID: {row['cve_id']}
        CVE Description: {row['cve_description']}

        Your response should be strictly in the following format:
        {component}:X  ← where X is one of the valid option keys.

        Do not include explanations or extra text.
    '''

def construct_vector_prompt(row):
    return f'''
        You are an expert in cybersecurity and vulnerability assessment. Given a CVE description, your task is to extract and generate the corresponding CVSS 3.1 vector in the standard format.

        Vector Components:
        - AV: Attack Vector [Network (N), Adjacent (A), Local (L), Physical (P)]
        - AC: Attack Complexity [Low (L), High (H)]
        - PR: Privileges Required [None (N), Low (L), High (H)]
        - UI: User Interaction [None (N), Required (R)]
        - S: Scope [Unchanged (U), Changed (C)]
        - C: Confidentiality [None (N), Low (L), High (H)]
        - I: Integrity [None (N), Low (L), High (H)]
        - A: Availability [None (N), Low (L), High (H)]

        Your task is to generate the CVSS vector for the following CVE:
        CVE ID: {row['cve_id']}
        CVE Description: {row['cve_description']}

        Your response should be strictly in the following format:
        CVSS:3.1/AV:X/AC:X/PR:X/UI:X/S:X/C:X/I:X/A:X  ← where X is one of the valid option keys.

        Do not include explanations, disclaimers, or extra text. Ensure the output strictly follows the CVSS 3.1 vector format.
    '''

def construct_vector_few_shot_prompt(row):
    return f'''
        You are an expert in cybersecurity and vulnerability assessment. Given a CVE description, your task is to extract and generate the corresponding CVSS 3.1 vector in the standard format.

        Examples:
        ---
        CVE ID: CVE-2002-20002
        CVE Description: The Net::EasyTCP package before 0.15 for Perl always uses Perl's builtin rand(), which is not a strong random number generator, for cryptographic keys.
        Vector:  CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:C/C:L/I:L/A:N
        ---
        CVE ID: CVE-2003-5001
        CVE Description: A vulnerability was found in ISS BlackICE PC Protection and classified as critical. Affected by this issue is the component Cross Site Scripting Detection. The manipulation as part of POST/PUT/DELETE/OPTIONS Request leads to privilege escalation. The attack may be launched remotely. The exploit has been disclosed to the public and may be used. It is recommended to upgrade the affected component. NOTE: This vulnerability only affects products that are no longer supported by the maintainer.
        Vector:  CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N
        ---
        CVE ID: CVE-2017-20011
        CVE Description: A vulnerability was found in WEKA INTEREST Security Scanner 1.8. It has been rated as problematic. This issue affects some unknown processing of the component HTTP Handler. The manipulation with an unknown input leads to denial of service. It is possible to launch the attack on the local host. The exploit has been disclosed to the public and may be used. NOTE: This vulnerability only affects products that are no longer supported by the maintainer.
        Vector:  CVSS:3.1/AV:L/AC:L/PR:L/UI:R/S:U/C:N/I:N/A:L
        ---

        Vector Components:
        - AV: Attack Vector [Network (N), Adjacent (A), Local (L), Physical (P)]
        - AC: Attack Complexity [Low (L), High (H)]
        - PR: Privileges Required [None (N), Low (L), High (H)]
        - UI: User Interaction [None (N), Required (R)]
        - S: Scope [Unchanged (U), Changed (C)]
        - C: Confidentiality [None (N), Low (L), High (H)]
        - I: Integrity [None (N), Low (L), High (H)]
        - A: Availability [None (N), Low (L), High (H)]

        Your task is to generate the CVSS vector for the following CVE:
        CVE ID: {row['cve_id']}
        CVE Description: {row['cve_description']}
        
        Your response should be strictly in the following format:
        CVSS:3.1/AV:X/AC:X/PR:X/UI:X/S:X/C:X/I:X/A:X  ← where X is one of the valid option keys.

        Do not include explanations, disclaimers, or extra text. Ensure the output strictly follows the CVSS 3.1 vector format.
    '''


def construct_cwe_vector_prompt(row):
    prompt = '''
        You are an expert in cybersecurity and vulnerability assessment. Given a CVE description and its corresponding CWE data, your task is to extract and generate the corresponding CVSS 3.1 vector in the standard format.

        Vector Components:
        - AV: Attack Vector [Network (N), Adjacent (A), Local (L), Physical (P)]
        - AC: Attack Complexity [Low (L), High (H)]
        - PR: Privileges Required [None (N), Low (L), High (H)]
        - UI: User Interaction [None (N), Required (R)]
        - S: Scope [Unchanged (U), Changed (C)]
        - C: Confidentiality [None (N), Low (L), High (H)]
        - I: Integrity [None (N), Low (L), High (H)]
        - A: Availability [None (N), Low (L), High (H)]

        Your task is to generate the CVSS vector for the following CVE:
    '''

    prompt += f"\nCVE ID: {row['cve_id']}"
    prompt += f"\nCVE Description: {row['cve_description']}"

    # Including CWE data only if =/= "N/A"
    if row.get("cwe_id") and row["cwe_id"] != "N/A":
        prompt += f"\n\nThe CVE leverages this CWE:\nCWE ID: {row['cwe_id']}"

        if row.get("cwe_description") and row["cwe_description"] != "N/A":
            prompt += f"\nCWE Description: {row['cwe_description']}"

        if row.get("cwe_enhanced_description") and row["cwe_enhanced_description"] != "N/A":
            prompt += f"\nCWE Enhanced Description: {row['cwe_enhanced_description']}"

        if row.get("cwe_consequences") and row["cwe_consequences"] != "N/A":
            prompt += f"\nCWE Consequences: {row['cwe_consequences']}"

        if row.get("cwe_mitigations") and row["cwe_mitigations"] != "N/A":
            prompt += f"\nCWE Mitigations: {row['cwe_mitigations']}"

    prompt += '''
        Your response should be strictly in the following format:
        CVSS:3.1/AV:X/AC:X/PR:X/UI:X/S:X/C:X/I:X/A:X  ← where X is one of the valid option keys.

        Do not include explanations, disclaimers, or extra text. Ensure the output strictly follows the CVSS 3.1 vector format.
    '''
    
    return prompt


def query_openai(model, prompt):
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            )

        return response.output_text
    except Exception as e:
        return "N/A"


def query_ollama(model, prompt):
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip("\"'")
    except Exception as e:
        return "N/A"



# Requires APIs
useOpenAI = True

# Dataset sampling
sample_size = 1000

# Defining models
ollama_models = [
    'llama3.2',
    'gemma3:12b',
    'phi4',
    'deepseek-r1:8b',
    'mistral',
    'qwq',
    'qwen2.5:14b',
]

openai_models = [
    'gpt-4o-mini',
    'gpt-4o',
    'gpt-3.5-turbo',
]

# CVSS Components
CVSS_COMPONENTS = {
    "AV": (
        {"N": "Network", "A": "Adjacent", "L": "Local", "P": "Physical"},
        "AV (Attack Vector): Defines how the vulnerability is exploited."),
    "AC": (
        {"L": "Low", "H": "High"},
        "AC (Attack Complexity): Describes the conditions beyond the attacker's control."),
    "PR": (
        {"N": "None", "L": "Low", "H": "High"},
        "PR (Privileges Required): What level of privileges the attacker must possess."),
    "UI": (
        {"N": "None", "R": "Required"},
        "UI (User Interaction): Whether a user must participate."),
    "S":  (
        {"U": "Unchanged", "C": "Changed"},
        "S (Scope): Whether other components are affected."),
    "C":  (
        {"N": "None", "L": "Low", "H": "High"},
        "C (Confidentiality): Impact on confidentiality."),
    "I":  (
        {"N": "None", "L": "Low", "H": "High"},
        "I (Integrity): Impact on integrity."),
    "A":  (
        {"N": "None", "L": "Low", "H": "High"},
        "A (Availability): Impact on availability."),
}

if useOpenAI:
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

# Create folder for LLM CSVs if it doesn't exist
results_folder = './results/llms'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


# Loading dataset
df = pd.read_parquet('./dataset.parquet')
df = df.tail(sample_size).sort_values(by='cve_id')

models = ollama_models if not useOpenAI else openai_models + ollama_models

for llm in models:
    # Create a folder for each model
    model_results_folder = os.path.join(results_folder, llm.replace(":", "_"))
    if not os.path.exists(model_results_folder):
        os.makedirs(model_results_folder)
    # Output file paths
    output_components = os.path.join(model_results_folder, f'{llm.replace(":", "_")}_components.csv')
    output_vector = os.path.join(model_results_folder, f'{llm.replace(":", "_")}_vector.csv')
    output_vector_few_shot = os.path.join(model_results_folder, f'{llm.replace(":", "_")}_vector_few_shot.csv')
    output_cwe_vector = os.path.join(model_results_folder, f'{llm.replace(":", "_")}_cwe_vector.csv')

    components_results = []
    vector_results = []
    vector_few_shot_results = []
    cwe_vector_results = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"{llm}"):
        # Constructing prompts
        component_prompts = {}
        for component, (options, description) in CVSS_COMPONENTS.items():
            prompt = construct_component_prompt(row, component, options, description)
            component_prompts[component] = prompt
        vector_prompt = construct_vector_prompt(row)
        vector_few_shot_prompt = construct_vector_few_shot_prompt(row)
        cwe_vector_prompt = construct_cwe_vector_prompt(row)

        # Querying Ollama models
        if llm in ollama_models:
            component_results = {component: query_ollama(llm, prompt) for component, prompt in component_prompts.items()}
            vector_result = query_ollama(llm, vector_prompt)
            vector_few_shot_result = query_ollama(llm, vector_few_shot_prompt)
            cwe_vector_result = query_ollama(llm, cwe_vector_prompt)
        # Querying OpenAI models
        elif llm in openai_models:
            component_results = {component: query_openai(llm, prompt) for component, prompt in component_prompts.items()}
            vector_result = query_openai(llm, vector_prompt)
            vector_few_shot_result = query_openai(llm, vector_few_shot_prompt)
            cwe_vector_result = query_openai(llm, cwe_vector_prompt)

        # Storing results
        components_results.append({
            'cve_id': row['cve_id'],
            **{f"{component}_response": component_results[component] for component in component_results}
        })
        vector_results.append({
            'cve_id': row['cve_id'],
            'response': vector_result
        })
        vector_few_shot_results.append({
            'cve_id': row['cve_id'],
            'response': vector_few_shot_result
        })
        cwe_vector_results.append({
            'cve_id': row['cve_id'],
            'response': cwe_vector_result
        })

        if index % 10 == 0:
            pd.DataFrame(components_results).to_csv(output_components, index=False)
            pd.DataFrame(vector_results).to_csv(output_vector, index=False)
            pd.DataFrame(vector_few_shot_results).to_csv(output_vector_few_shot, index=False)
            pd.DataFrame(cwe_vector_results).to_csv(output_cwe_vector, index=False)

    pd.DataFrame(components_results).to_csv(output_components, index=False)
    pd.DataFrame(vector_results).to_csv(output_vector, index=False)
    pd.DataFrame(vector_few_shot_results).to_csv(output_vector_few_shot, index=False)
    pd.DataFrame(cwe_vector_results).to_csv(output_cwe_vector, index=False)
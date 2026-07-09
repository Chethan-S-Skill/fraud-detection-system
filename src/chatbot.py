import re

class ShieldAICopilot:
    """
    NLP Security Copilot Router.
    Parses conversational commands from security operators and maps them
    to context-rich explanations and direct frontend actions.
    """
    def __init__(self):
        pass

    def respond(self, message: str) -> dict:
        msg = message.strip().lower()
        
        # 1. Intent: Navigate / Tab Switching
        tab_patterns = {
            'dashboard': ['dashboard', 'soc', 'real-time', 'ledger', 'live feed', 'transactions'],
            'rules': ['rule', 'policy', 'policies', 'rules engine', 'engine', 'deploy rule'],
            'model': ['model', 'analytics', 'diagnostics', 'auc', 'roc', 'f1', 'confusion matrix', 'importance'],
            'monitoring': ['monitor', 'monitoring', 'prometheus', 'telemetry', 'grafana', 'drift', 'latency']
        }
        
        for tab, keywords in tab_patterns.items():
            if any(kw in msg for kw in keywords) and ('go' in msg or 'switch' in msg or 'show' in msg or 'take' in msg or 'open' in msg):
                friendly_names = {
                    'dashboard': 'Real-Time SOC Dashboard',
                    'rules': 'Rule Engine Policy Panel',
                    'model': 'Model Diagnostics tab',
                    'monitoring': 'MLOps Monitoring Telemetry'
                }
                return {
                    "text": f"Switching your workspace to the **{friendly_names[tab]}**.",
                    "action": "SWITCH_TAB",
                    "target": tab
                }

        # 2. Intent: Investigate / Explain Transaction ID
        txn_match = re.search(r'txn_\d{7}', msg)
        if txn_match:
            txn_id = txn_match.group(0).upper()
            return {
                "text": f"Locating transaction records for **{txn_id}** and generating SHAP explainability matrices... Opening forensic investigation panel now.",
                "action": "OPEN_FORENSIC",
                "target": txn_id
            }
        
        if 'explain last' in msg or 'explain latest' in msg or 'analyze last' in msg or 'check last' in msg:
            return {
                "text": "Fetching forensic details and local SHAP force contributions for the latest transaction...",
                "action": "OPEN_FORENSIC",
                "target": "LATEST"
            }

        # 3. Intent: Adjust Model Threshold
        threshold_match = re.search(r'(?:threshold|sensitivity|boundary)(?:\s+to)?\s*(0\.\d+|1\.0|\d+%)', msg)
        if threshold_match:
            val_str = threshold_match.group(1)
            val = 0.5
            if '%' in val_str:
                val = float(val_str.replace('%', '')) / 100.0
            else:
                val = float(val_str)
                
            val = min(max(val, 0.0), 1.0)
            return {
                "text": f"Adjusting classifier decision threshold boundary to **{val:.2f}**. Updating operations metrics and Confusion Matrix calculations.",
                "action": "SET_THRESHOLD",
                "value": val
            }

        # 4. Intent: Programmatic Rule Deployment
        if 'create' in msg or 'deploy' in msg or 'add' in msg or 'block' in msg or 'flag' in msg:
            # Check fields
            field = None
            if 'amount' in msg or 'spend' in msg or 'money' in msg or 'value' in msg or '$' in msg:
                field = "amount"
            elif 'distance' in msg or 'far' in msg or 'km' in msg or 'miles' in msg:
                field = "distance_from_home"
            elif 'count' in msg or 'frequency' in msg or 'velocity' in msg:
                field = "txn_count_1h"
            
            # Check operator
            op = ">"
            if 'less' in msg or 'under' in msg or 'below' in msg:
                op = "<"
            
            # Check action
            action = "Block"
            if 'flag' in msg or 'review' in msg:
                action = "Flag"
                
            # Extract number
            numbers = re.findall(r'\d+', msg)
            if field and numbers:
                val = float(numbers[0])
                rule_name = f"COPILOT_AUTO_{field.upper()}"
                return {
                    "text": f"Deploying programmatic security policy: **{rule_name}** (Trigger: {field} {op} {val} -> {action}).",
                    "action": "CREATE_RULE",
                    "rule": {
                        "name": rule_name,
                        "field": field,
                        "op": op,
                        "value": val,
                        "action": action
                    }
                }

        # 5. Intent: Default Help Responses
        if 'help' in msg or 'what can' in msg or 'commands' in msg:
            return {
                "text": (
                    "I am the ShieldAI MLOps Assistant. You can run conversational commands like:\n\n"
                    "- **Switch screens**: 'go to rules', 'show model performance'\n"
                    "- **Investigate fraud**: 'explain latest transaction', 'analyze TXN_7482931'\n"
                    "- **Tweak parameters**: 'set threshold to 0.65', 'lower sensitivity to 40%'\n"
                    "- **Create rules**: 'block transactions where amount > 5000', 'flag distance > 1000'"
                )
            }
            
        if 'hello' in msg or 'hi' in msg or 'hey' in msg:
            return {
                "text": "Hello, Operator. I am ShieldAI Copilot. How can I assist you in monitoring fraud metrics or executing policies today?"
            }
            
        if 'status' in msg or 'system' in msg or 'health' in msg:
            return {
                "text": "All systems operating normally. Model server loaded: XGBoost v1.5, preprocessors online. Prometheus scrapes at 10s intervals."
            }

        # Generic Fallback Response
        return {
            "text": "I received your query. If you'd like to investigate a transaction, write 'explain TXN_XXXXXXX'. To navigate, write 'switch to [tab]'. Type 'help' to see more options."
        }

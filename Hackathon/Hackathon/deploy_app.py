import requests
import time
import json

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
SOURCE_PATH = "/Workspace/Users/hs22h059@smail.iitm.ac.in/Hackathon"
APP_NAME = "ipo-forensics-ai"

# COMMAND ----------

# create app
create_resp = requests.post(
    f"https://{host}/api/2.0/apps",
    headers=headers,
    json={
        "name": APP_NAME,
        "description": "IPO forensics RAG app powered by Sarvam-m and Databricks Vector Search"
    }
)
print("create:", create_resp.status_code, create_resp.json().get("compute_status", {}).get("state"))

# COMMAND ----------

# wait for RUNNING then deploy
for i in range(20):
    status = requests.get(f"https://{host}/api/2.0/apps/{APP_NAME}", headers=headers).json()
    state = status.get("compute_status", {}).get("state")
    print(f"[{i*15}s] state: {state}")
    if state == "RUNNING":
        deploy_resp = requests.post(
            f"https://{host}/api/2.0/apps/{APP_NAME}/deployments",
            headers=headers,
            json={"source_code_path": SOURCE_PATH}
        )
        print("deploy:", deploy_resp.status_code, deploy_resp.json())
        break
    time.sleep(15)

# COMMAND ----------

# monitor deployment and print final URL
for i in range(40):
    status = requests.get(f"https://{host}/api/2.0/apps/{APP_NAME}", headers=headers).json()
    state = status.get("compute_status", {}).get("state")
    url = status.get("url", "not available yet")
    print(f"[{i*15}s] state: {state} | url: {url}")
    if state == "ACTIVE":
        print(f"app is live at: {url}")
        break
    time.sleep(15)

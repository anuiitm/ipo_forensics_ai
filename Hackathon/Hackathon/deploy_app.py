# Databricks notebook source
import requests, json

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

create_resp = requests.post(
    f"https://{host}/api/2.0/apps",
    headers=headers,
    json={
        "name": "ipo-forensics-ai",
        "description": "IPO forensics RAG app Databricks Vector Search"
    }
)
print("create:", create_resp.status_code, create_resp.json())

deploy_resp = requests.post(
    f"https://{host}/api/2.0/apps/ipo-forensics-ai/deployments",
    headers=headers,
    json={"source_code_path": "/Workspace/Users/hs22h059@smail.iitm.ac.in/Hackathon"}
)
print("deploy:", deploy_resp.status_code, deploy_resp.json())

# COMMAND ----------

print(host)

# COMMAND ----------

import time

for i in range(20):
    status_resp = requests.get(
        f"https://{host}/api/2.0/apps/ipo-forensics-ai",
        headers=headers
    )
    data = status_resp.json()
    state = data.get("compute_status", {}).get("state")
    print(f"[{i*15}s] state: {state}")
    if state == "ACTIVE":
        # now deploy
        deploy_resp = requests.post(
            f"https://{host}/api/2.0/apps/ipo-forensics-ai/deployments",
            headers=headers,
            json={"source_code_path": "/Workspace/Users/hs22h059@smail.iitm.ac.in/Hackathon"}
        )
        print("deploy:", deploy_resp.status_code, deploy_resp.json())
        break
    time.sleep(15)

# COMMAND ----------

import time

for i in range(20):
    status_resp = requests.get(
        f"https://{host}/api/2.0/apps/ipo-forensics-ai",
        headers=headers
    )
    data = status_resp.json()
    state = data.get("compute_status", {}).get("state")
    url = data.get("url", "not available yet")
    print(f"[{i*15}s] state: {state} | url: {url}")
    if state == "ACTIVE":
        print(f"app is live at: {url}")
        break
    time.sleep(15)

# COMMAND ----------

# DBTITLE 1,Redeploy app with fixes
import requests

# Redeploy with fixed code
deploy_resp = requests.post(
    f"https://{host}/api/2.0/apps/ipo-forensics-ai/deployments",
    headers=headers,
    json={"source_code_path": "/Workspace/Users/hs22h059@smail.iitm.ac.in/Hackathon"}
)
print("Redeploy status:", deploy_resp.status_code)
print(deploy_resp.json())

# COMMAND ----------

# DBTITLE 1,Monitor deployment status
import time

deployment_id = "01f1407691961516814cdaf15545a909"  # Latest deployment

print("Monitoring deployment...\n")
for i in range(40):  # 10 minutes max
    deploy_resp = requests.get(
        f"https://{host}/api/2.0/apps/ipo-forensics-ai/deployments/{deployment_id}",
        headers=headers
    )
    
    data = deploy_resp.json()
    state = data.get('status', {}).get('state')
    message = data.get('status', {}).get('message', '')
    
    print(f"[{i*15}s] {state}: {message}")
    
    if state == "ACTIVE":
        app_resp = requests.get(
            f"https://{host}/api/2.0/apps/ipo-forensics-ai",
            headers=headers
        )
        url = app_resp.json().get('url', 'N/A')
        print(f"\n🎉 SUCCESS! App is live at:\n{url}")
        break
    elif state == "FAILED":
        print(f"\n❌ FAILED: {message}")
        print("Check logs at: https://dbc-769a5f05-e643.cloud.databricks.com/#apps/ipo-forensics-ai")
        break
    
    time.sleep(15)

# COMMAND ----------

# DBTITLE 1,Get app URL
# Get app status and URL
app_resp = requests.get(
    f"https://{host}/api/2.0/apps/ipo-forensics-ai",
    headers=headers
)

app_data = app_resp.json()
url = app_data.get('url', 'N/A')
compute_state = app_data.get('compute_status', {}).get('state')

print(f"✅ App Status: {compute_state}")
print(f"🎉 App URL: {url}")
print(f"\nYour IPO Forensics AI app is now live!")

# COMMAND ----------

# DBTITLE 1,Check deployment status
# Quick status check
deployment_id = "01f1407691961516814cdaf15545a909"

deploy_resp = requests.get(
    f"https://{host}/api/2.0/apps/ipo-forensics-ai/deployments/{deployment_id}",
    headers=headers
)

data = deploy_resp.json()
print(json.dumps(data, indent=2))
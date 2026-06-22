from client import execute

status, body = execute(
    "createProject",
    {
        "id": "Testing123",
        "name": "ai_test1",
        "title": "AI Test1",
        "globalConfig": {},
        "isActive": True,
        "projectPackages": []
    }
)

print(status)
print(body)
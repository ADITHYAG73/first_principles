# Re-run pass 1 after the SSM-timeout fix, reusing the 152 finished generations of attempt 1-a5 (same rids + seeds).
req1 = json.load(open("work/req_pass1.json")); req1["resume_from"] = "1-a5"
json.dump(req1, open("work/req_pass1.json", "w")); print({k: v for k, v in req1.items() if k != "problems"})

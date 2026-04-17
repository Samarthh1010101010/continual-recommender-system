import torch
import os

task_file = "fixed_tasks/task_0.pt"
if os.path.exists(task_file):
    task = torch.load(task_file, weights_only=True)
    test_data = task["test"]
    
    user_positives = {}
    for u, label in zip(test_data["user"].tolist(), test_data["label"].tolist()):
        if float(label) > 0.0:
            user_positives[u] = user_positives.get(u, 0) + 1
    
    multi_pos = sum(1 for count in user_positives.values() if count > 1)
    total_pos_users = len(user_positives)
    
    print("Test data shape:", len(test_data['user']))
    print("Users with positives:", total_pos_users)
    print("Users with multiple positives:", multi_pos)
    if multi_pos > 0:
        counts = sorted([c for c in user_positives.values() if c > 1])
        print("Multi-positive distribution:", counts[:10])
else:
    print("Task file not found")

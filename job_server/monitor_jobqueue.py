from job_server import JobClient

client = JobClient()

# Check server health
health = client.health_check()
print(f"Server status:    {health['status']}")
print(f"Pending jobs:     {health['pending_jobs']}")
print(f"Running jobs:     {health['running_jobs']}")

# View current queue (pending and running jobs)
client.print_queue()


# Monitor recent job history

import time
from IPython.display import clear_output

prev_output = ''

while True:
    history = client.get_history(limit=10)

    output = ''

    for job in history:
        job_id = job['job_id']
        result = client.get_status(job_id)
        created = result.created_at.strftime('%Y-%m-%d %H:%M:%S') if result.created_at else 'N/A'
        completed = result.completed_at.strftime('%H:%M:%S') if result.completed_at else '...'
        output += f"{job_id} {job['user']}\t{job['status']:10s}  {created} -> {completed}  {job['experiment_class']}\t{result.data_file_path}\n"

    if output != prev_output:
        clear_output(wait=True)
        print(f"Recent Job History (last change in status: {time.strftime('%Y-%m-%d %H:%M:%S')})")
        print("-" * 80)
        print(output)
        prev_output = output
    time.sleep(5)
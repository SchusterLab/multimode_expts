import sys
import time

from job_server import JobClient

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10

client = JobClient()

# Check server health
health = client.health_check()
print(f"Server status:    {health['status']}")
print(f"Pending jobs:     {health['pending_jobs']}")
print(f"Running jobs:     {health['running_jobs']}")

# View current queue (pending and running jobs)
client.print_queue()


# Monitor recent job history

prev_output = ''
prev_lines = 0

while True:
    history = client.get_history(limit=limit)

    output = ''

    for job in history:
        job_id = job['job_id']
        result = client.get_status(job_id)
        output += f"{job_id} {job['user']}\t{job['status']:10s}  {job['experiment_class']}\t{result.data_file_path}\n"

    if output != prev_output:
        # Move cursor up to overwrite previous output
        if prev_lines > 0:
            sys.stdout.write(f"\033[{prev_lines}A\033[J")

        header = f"Recent Job History (last change in status: {time.strftime('%Y-%m-%d %H:%M:%S')})\n"
        header += "-" * 80 + "\n"
        full_output = header + output
        sys.stdout.write(full_output)
        sys.stdout.flush()

        prev_lines = full_output.count('\n')
        prev_output = output
    time.sleep(5)
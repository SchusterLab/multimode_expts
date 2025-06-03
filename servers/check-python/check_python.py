import subprocess
import sys

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ImageContent, BlobResourceContents
import logging
import os

# Set up logging (this just prints messages to your terminal for debugging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the MCP server object
mcp = FastMCP()

def activate_slab_env():
    # Only Windows support
    cmd = 'conda activate slab && python -c "import sys; print(\'slab environment activated\'); print(sys.version)"'
    shell = True

    try:
        result = subprocess.run(cmd, shell=shell, check=True, capture_output=True, text=True)
        print("Activation output:\n", result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print("Failed to activate 'slab' environment:", e)
        print("Output:", e.output)
        sys.exit(1)
def activate_slab_env_and_import_qick():
    # Only Windows support
    # First command: activate slab and print version
    cmd1 = 'conda activate slab && python -c "import sys; print(\'slab environment activated\'); print(sys.version)"'
    # Second command: import qick
    cmd2 = 'conda activate slab && python -c "import qick; print(\'qick imported\')"'
    shell = True

    try:
        result1 = subprocess.run(cmd1, shell=shell, check=True, capture_output=True, text=True)
        print("Activation output:\n", result1.stdout.strip())
        result2 = subprocess.run(cmd2, shell=shell, check=True, capture_output=True, text=True)
        print("Qick import output:\n", result2.stdout.strip())
    except subprocess.CalledProcessError as e:
        print("Failed to activate 'slab' environment or import 'qick':", e)
        print("Output:", e.output)
        sys.exit(1)

@mcp.tool(
    name="single_shot",
    description="Performs a single shot measurement using the slab environment.",
)
def single_shot() -> None:
    """
    Performs a single shot measurement by activating the slab environment and running a measurement script.
    """
    measurement_script = os.path.join(os.path.dirname(__file__), "server.py")
    cmd = f'conda activate slab && python "{measurement_script}"'
    shell = True

    try:
        result = subprocess.run(cmd, shell=shell, check=True, capture_output=True, text=True)
        output = result.stdout.strip()
        logger.info("Single shot output: %s", output)
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to perform single shot: {e}\nOutput: {e.output}"
        logger.error(error_msg)
        
def run_server_in_slab_env():
    # Only Windows support
    # Activate slab environment and run server.py
    server_script = os.path.join(os.path.dirname(__file__), "server.py")
    cmd = f'conda activate slab && python "{server_script}"'
    shell = True

    try:
        result = subprocess.run(cmd, shell=shell, check=True, capture_output=True, text=True)
        print("Server output:\n", result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print("Failed to run server.py in 'slab' environment:", e)
        print("Output:", e.output)
        sys.exit(1)

def main():
    logger.info('Starting your-new-server')
    mcp.run('stdio')


if __name__ == "__main__":
    # activate_slab_env()
    # try_import_qick()
    # activate_slab_env_and_import_qick()
    # result = run_server_in_slab_env()
    # print('Server started successfully:', result)
    # check_env_and_import()
    # run_server_in_slab_env()
    main()

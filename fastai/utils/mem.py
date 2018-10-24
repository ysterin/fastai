# python -c "import fastai; print(fastai.utils.collect_env.get_gpu_mem(), fastai.utils.collect_env.get_gpu_with_max_free_mem())"
# [[495, 7624, 8119]] (0, 7624)

from enum import IntEnum
Memory = IntEnum('Memory', "USED, FREE, TOTAL", start=0)

# returns a list of mem available for each cpu
# [ [used-0, free-0, total-0], [used-1, free-1, total-1] ]
# this function assumes nvidia-smi works and will return [] if this is not the case
def get_gpu_mem():
    "query nvidia-smi for used, free and total memory for each available gpu"
    import subprocess

    mem = []
    try:
        cmd = "nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,nounits,noheader"
        result = subprocess.run(cmd.split(), shell=False, check=False, stdout=subprocess.PIPE)
    except: pass
    else:
        if result.returncode == 0 and result.stdout:
            output = result.stdout.decode('utf-8')
            mem = [[int(y) for y in x.split(', ')] for x in output.strip().split('\n') ]
            #print(mem)
    return mem

# return the gpu number that has the most memory, and the free memory
# return [] if no gpus were found
def get_gpu_with_max_free_mem():
    mem = np.array(get_gpu_mem())
    if not len(mem): return []
    id = np.argmax(mem[:,Memory.FREE])
    return (id, mem[id,Memory.FREE])



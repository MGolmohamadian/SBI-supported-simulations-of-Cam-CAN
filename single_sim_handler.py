import sys
import os
import time
import fcntl
import montbrio_simulation
import traceback
from multiprocessing import Process, Queue
import queue  # for handling queue.Empty exception

def func(q, subject, noise, G, dt, sim_len, weights_file_pattern, FCD_file_pattern):
    output = montbrio_simulation.process_sub(subject, noise, G, dt, sim_len, weights_file_pattern, FCD_file_pattern)
    q.put(output)

def process_simulation(subject, noise, G, dt, sim_len, time_per_sim, job_ID, job_array_num, weights_file_pattern, results_file_pattern, FCD_file_pattern):
    try:
        q = Queue()
        p = Process(target=func, args=(q, subject, noise, G, dt, sim_len, weights_file_pattern, FCD_file_pattern))
        p.start()
        p.join(time_per_sim)

        if p.is_alive():
            handle_timeout(p, subject, noise, G, dt, sim_len, time_per_sim, job_ID, job_array_num, weights_file_pattern, results_file_pattern, FCD_file_pattern)
        else:
            handle_completion(q, subject, noise, G, dt, sim_len, time_per_sim, job_ID, job_array_num, weights_file_pattern, results_file_pattern, FCD_file_pattern)

    except Exception as e:
        handle_exception(e, subject, noise, G, dt, sim_len, time_per_sim, job_ID, job_array_num, weights_file_pattern, results_file_pattern, FCD_file_pattern)

def handle_timeout(p, subject, noise, G, dt, sim_len, time_per_sim, job_ID, job_array_num, weights_file_pattern, results_file_pattern, FCD_file_pattern):
    print(f"Timeout: subject={subject} noise={noise} G={G} dt={dt} did not finish in time={timer_per_sim} seconds.")
    p.terminate()
    p.join()
    write_to_file(subject, noise, G, dt, sim_len, time_per_sim, job_ID, job_array_num, weights_file_pattern, results_file_pattern, FCD_file_pattern, "Timeout")

def handle_completion(q, subject, noise, G, dt, sim_len, time_per_sim, job_ID, job_array_num, weights_file_pattern, results_file_pattern, FCD_file_pattern):
    try:
        result = q.get_nowait()
        print(f"Completed: subject={subject} noise={noise} G={G} dt={dt}, Result={result}")
        write_to_file(subject, noise, G, dt, sim_len, time_per_sim, job_ID, job_array_num, weights_file_pattern, results_file_pattern, FCD_file_pattern, "Success", result)
    except queue.Empty:
        print(f"No result: subject={subject} noise={noise} G={G} dt={dt} did not return output.")
        write_to_file(subject, noise, G, dt, sim_len, time_per_sim, job_ID, job_array_num, weights_file_pattern, results_file_pattern, FCD_file_pattern, "No Result")

def handle_exception(e, subject, noise, G, dt, sim_len, time_per_sim, job_ID, job_array_num, weights_file_pattern, results_file_pattern, FCD_file_pattern):
    print(f"Error: subject={subject} noise={noise} G={G} dt={dt} raised an exception. {e}")
    traceback.print_exc()
    write_to_file(subject, noise, G, dt, sim_len, time_per_sim, job_ID, job_array_num, weights_file_pattern, results_file_pattern, FCD_file_pattern, "Error")

def write_to_file(subject, noise, G, dt, sim_len, time_per_sim, job_ID, job_array_num, weights_file_pattern, results_file_pattern, FCD_file_pattern, status, result=None):
    result = result or ["nan"] * 2  # Assume 2 outputs are expected from montbrio_simulation
    status_code = {"Success": 1, "Timeout": 2, "No Result": 3, "Error": 4}.get(status, 0)
    with open(results_file_pattern.format(subject=subject), 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(f"{subject}\t{noise}\t{G}\t{dt}\t{sim_len}\t{time_per_sim}\t{job_ID}\t{job_array_num}\t{result[0]}\t{result[1]}\t{result[2]}\t{result[3]}\t{status_code}\n")
        fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    
    subject = sys.argv[1]
    noise = float(sys.argv[2])
    G = float(sys.argv[3])
    dt = float(sys.argv[4])
    sim_len = float(sys.argv[5])
    time_per_sim = int(sys.argv[6])
    job_ID = sys.argv[7]
    job_array_num = sys.argv[8]
    weights_file_pattern = sys.argv[9]
    results_file_pattern = sys.argv[10]
    FCD_file_pattern = sys.argv[11]

    process_simulation(subject, noise, G, dt, sim_len, time_per_sim, job_ID, job_array_num, weights_file_pattern, results_file_pattern, FCD_file_pattern)

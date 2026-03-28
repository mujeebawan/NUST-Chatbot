"""Benchmark the chatbot's performance: speed, RAM usage, accuracy."""
import time
import os
import psutil
from rag import NUSTAdmissionsBot

TEST_QUESTIONS = [
    "What is the NET exam?",
    "How is the merit calculated for engineering programs?",
    "What programs does SEECS offer?",
    "Can A-Level students apply to NUST?",
    "What are the hostel facilities at NUST?",
    "What is the fee structure for engineering?",
    "How many times can I attempt NET?",
    "What documents are needed for admission?",
    "Is there negative marking in NET?",
    "What scholarships does NUST offer?",
]


def get_ram_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_benchmark():
    print("=" * 60)
    print("NUST Admissions Chatbot - Benchmark Report")
    print("=" * 60)

    # System info
    print(f"\nSystem: {os.cpu_count()} cores, "
          f"{psutil.virtual_memory().total / (1024**3):.1f} GB total RAM")
    print(f"RAM before loading: {get_ram_usage_mb():.0f} MB")

    # Load bot
    print("\n--- Loading Model ---")
    load_start = time.time()
    bot = NUSTAdmissionsBot()
    bot.load()
    load_time = time.time() - load_start
    print(f"Load time: {load_time:.1f}s")
    print(f"RAM after loading: {get_ram_usage_mb():.0f} MB")

    # Run test questions
    print("\n--- Inference Benchmark ---")
    times = []
    for i, question in enumerate(TEST_QUESTIONS, 1):
        start = time.time()
        result = bot.ask(question)
        elapsed = time.time() - start
        times.append(elapsed)

        answer_preview = result["answer"][:100] + "..." if len(result["answer"]) > 100 else result["answer"]
        print(f"\nQ{i}: {question}")
        print(f"A: {answer_preview}")
        print(f"Time: {elapsed:.1f}s | Sources: {', '.join(result['sources'])}")

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Model load time: {load_time:.1f}s")
    print(f"Peak RAM usage: {get_ram_usage_mb():.0f} MB")
    print(f"Average response time: {sum(times)/len(times):.1f}s")
    print(f"Fastest response: {min(times):.1f}s")
    print(f"Slowest response: {max(times):.1f}s")
    print(f"Total questions: {len(times)}")
    print(f"Questions answered: {sum(1 for t in times if t > 0)}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()

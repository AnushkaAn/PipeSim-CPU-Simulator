import random
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from io import BytesIO

# --- Cache Class ---
class Cache:
    def __init__(self, size_kb, block_size, pipeline_stages, name):
        self.size_kb = size_kb
        self.block_size = block_size
        self.pipeline_stages = pipeline_stages
        self.name = name
        self.cache_blocks = size_kb * 1024 // block_size
        self.hits = 0
        self.misses = 0

    def access(self, address):
        if random.random() < 0.8:
            self.hits += 1
            return True
        else:
            self.misses += 1
            return False

    def miss_rate(self):
        total_accesses = self.hits + self.misses
        return self.misses / total_accesses if total_accesses > 0 else 0

# --- Branch Predictor Class ---
class BranchPredictor:
    def __init__(self):
        self.predictions = {"correct": 0, "incorrect": 0}

    def predict(self, is_taken):
        if random.random() < 0.7:
            self.predictions["correct"] += 1
            return is_taken
        else:
            self.predictions["incorrect"] += 1
            return not is_taken

    def accuracy(self):
        total = self.predictions["correct"] + self.predictions["incorrect"]
        return self.predictions["correct"] / total if total > 0 else 0

# --- Pipeline Class ---
class Pipeline:
    def __init__(self, stages, l1_cache, branch_predictor=None):
        self.stages = stages
        self.l1_cache = l1_cache
        self.branch_predictor = branch_predictor
        self.cycles = 0
        self.instructions = 0
        self.branch_penalty = 5

    def execute_instruction(self, address, is_branch=False, branch_taken=False):
        self.instructions += 1
        self.cycles += self.stages

        if not self.l1_cache.access(address):
            self.cycles += 10

        if is_branch and self.branch_predictor:
            predicted_taken = self.branch_predictor.predict(branch_taken)
            if predicted_taken != branch_taken:
                self.cycles += self.branch_penalty

    def cpi(self):
        return self.cycles / self.instructions if self.instructions > 0 else 0

# --- Simulation Function ---
def simulate(cache_size, block_size, pipeline_stages, num_instructions=1000, use_branch_prediction=False):
    l1_cache = Cache(cache_size, block_size, pipeline_stages, "L1 Cache")
    branch_predictor = BranchPredictor() if use_branch_prediction else None
    pipeline = Pipeline(pipeline_stages, l1_cache, branch_predictor)

    for _ in range(num_instructions):
        address = random.randint(0, l1_cache.cache_blocks - 1)
        is_branch = random.random() < 0.2
        branch_taken = random.random() < 0.5
        pipeline.execute_instruction(address, is_branch, branch_taken)

    return {
        "Cache Size (KB)": cache_size,
        "Pipeline Stages": pipeline_stages,
        "CPI": pipeline.cpi(),
        "Miss Rate": l1_cache.miss_rate(),
        "Branch Accuracy": branch_predictor.accuracy() if branch_predictor else None
    }

# --- Plot Function ---
def plot_simulation_results(results, metric):
    fig, ax = plt.subplots(figsize=(10, 5))
    for stage in sorted(set([r["Pipeline Stages"] for r in results])):
        filtered = [r for r in results if r["Pipeline Stages"] == stage]
        sizes = [r["Cache Size (KB)"] for r in filtered]
        values = [r[metric] for r in filtered]
        ax.plot(sizes, values, marker="o", label=f"{stage} Stages")

    ax.set_title(f"{metric} vs Cache Size")
    ax.set_xlabel("Cache Size (KB)")
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# --- Streamlit UI ---
st.title("🚀 PipeSim - Pipelined Cache Simulation")
st.markdown("""
This tool simulates CPU performance by modeling **cache behavior**, **pipeline stages**, and **branch prediction**.
Use the controls to tweak simulation parameters and view CPI and cache performance.
""")

# Input controls
cache_sizes = st.multiselect("Select Cache Sizes (KB):", [4, 8, 16, 32, 64], default=[8, 16])
pipeline_stages_list = st.multiselect("Select Pipeline Stages:", [1, 2, 3, 4, 5], default=[1, 2])
num_instructions = st.slider("Number of Instructions:", 100, 5000, 1000, step=100)
branch_prediction = st.checkbox("Enable Branch Prediction", value=True)

# Run simulation
if st.button("🔁 Run Simulation"):
    simulation_results = []

    for stages in pipeline_stages_list:
        for size in cache_sizes:
            result = simulate(
                cache_size=size,
                block_size=16,
                pipeline_stages=stages,
                num_instructions=num_instructions,
                use_branch_prediction=branch_prediction
            )
            simulation_results.append(result)

    # ✅ Show Summary Table
    df = pd.DataFrame(simulation_results)
    st.subheader("📊 Simulation Summary Table")
    st.dataframe(df.style.format({
        "CPI": "{:.2f}",
        "Miss Rate": "{:.2%}",
        "Branch Accuracy": "{:.2%}"
    }))

    # ✅ Show Graphs
    st.subheader("📈 CPI vs Cache Size")
    plot_simulation_results(simulation_results, "CPI")

    st.subheader("📉 Miss Rate vs Cache Size")
    plot_simulation_results(simulation_results, "Miss Rate")

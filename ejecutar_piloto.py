import random

from global_values import (
    LLM_NAME,
    MULPROCESS_CNT,
    SAMPLE_NUM,
)
from scripts.experiment import Experiment


def main():
    random.seed(42)

    print("Modelo:", LLM_NAME)
    print("Preguntas por variante:", SAMPLE_NUM)

    experiment = Experiment(
        mulprocess_cnt=MULPROCESS_CNT,
        llm_name=LLM_NAME,
    )

    print("\nEjecutando NL2SQL...")
    experiment.run_nl2sql_baseline(sample_num=SAMPLE_NUM)

    print("\nEjecutando AutoPrep + NL2SQL...")
    experiment.run_multi_agents_data_prep(sample_num=SAMPLE_NUM)

    print("\nLas dos evaluaciones finalizaron.")


if __name__ == "__main__":
    main()
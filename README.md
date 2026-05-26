# air-quality-forecasting
Exam project for the course 'Data Science, Prediction and Forecasting' @ Aarhus University, 2026


## Project Structure

```
monopolet-nlp/
│
├── data/
│   └── aarhus_air_quality.csv           # 
│
├── nbs/                                 # Jupyter Notebooks
│   ├── monopolet_cleaning.ipynb         # 
│   ├── postprocessing_llm_answers.ipynb # 
│   ├── postprocessing_scores.ipynb      # 
│   └── combine_and_add_topics.ipynb     #     
│
├── src/                                 # Source code (contains main analyses)
│   ├── prompting_gemma.py               # 
│   ├── prompting_llama.py               # 
│   ├── cosine_similarity.py             # 
│   ├── llm_as_a_judge.py                # 
│   ├── topics_to_dummy_data.py          # 
│   ├── glmm.R                           # 
│   └── violinplots.R                    # 
│
├── plots/                               # plots
│   ├── violin_cosine.png                # violin plot for cosine similarity
│   ├── violin_llm_judge.png             # violin plot for llm-as-a-judge
│   ├── violin_cosine_DUMMY.png          # violin plot for cosine similarity for dummy data
│   └── violin_llm_judge_DUMMY.png       # violin plot for llm-as-a-judge for dummy data
│
├── .env                                 # contains HuggingFace token (currently empty, needs to be specified by the user)
├── README.md                    
├── requirements.txt                     # Python dependencies
├── run_python_scripts_dummy.sh          # run python scripts from src/ with dummy data on GPU
├── run_python_scripts_dummy_CPU.sh      # run python scripts from src/ with dummy data on CPU
├── run_r_scripts_dummy.sh               # Run R scripts from src/ with set dummy data
├── env_to_jupyter.sh                    # creates kernel from .venv to be used for jupyter 
└── setup.sh                             # Set up virtual environment and install required packages
```
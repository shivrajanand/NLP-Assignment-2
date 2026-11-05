# NLP Assignment 2

## Environment Setup

The repository includes three environment files for dependency installation and reproducibility:

- `requirements.txt`: Pinned Python dependencies.
- `explicit.txt`: Complete environment snapshot with installed package versions.
- `environment.yml`: Conda environment specification.

### Option 1: Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate assignment
```

### Option 2: Using pip

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Project Details

**Dataset:** WikiBio GPT-3 Hallucination Dataset

**Group Members:**
1. **Shibhaditya Dohare** | shibhadityadohare@gmail.com | `22EE38014`
2. **Shivraj Anand** | shivrajanand022002@gmail.com | `25CS92P07`
3. **Avinash** | avinashg2312@gmail.com | `24CS10009`

# Project Structure

| File/Folder | Description |
|---|---|
| `analysis/` | Contains individual analysis files for all tasks. |
| `data/` | Contains the assigned raw dataset (`raw_hf`) and other preprocessed datasets. |
| `results/` | Stores the results of all tasks, organized by task. |
| `Tasks&Data/` | Contains the task descriptions and datasets provided in the instructor's PDFs. |
| `data.ipynb` | Jupyter notebook for data preprocessing and preparation. |
| `task1.ipynb` | Jupyter notebook containing the implementation of Task 1. |
| `task2.ipynb` | Jupyter notebook containing the implementation of Task 2. |
| `task3.ipynb` | Jupyter notebook containing the implementation of Task 3. |
| `environment.yml` | Conda environment specification for reproducing the project environment. |
| `requirements-lock.txt` | Complete list of installed Python libraries with pinned versions. |
| `requirements.txt` | Essential Python dependencies required to run the project. |
| `REPORT.md` | A basic report on all the experiments done |


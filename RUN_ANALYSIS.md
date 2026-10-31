# Run the capstone analysis

The official course dataset is included at `data/dataset_part_2.csv`.

1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python analysis.py`
3. Open the `evidence/` folder for EDA charts, SQL result CSVs, the Folium HTML map, model metrics, model comparison chart, confusion matrix, and `results_summary.json`.

The map uses actual launch-site coordinates from the dataset. The 10 km circles are illustrative proximity buffers, not actual landing/recovery tracks. Model evaluation is based on a stratified hold-out split; metrics can vary with data and environment versions.

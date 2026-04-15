# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = r"/Users/vinodtonda/Downloads/RagPipelineProject/myenv/source/nltestdata.csv"


path = kagglehub.dataset_download("frankossai/natural-questions-dataset")
print(path)

# Load the latest version
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "frankossai/natural-questions-dataset",
  file_path,
)

print("First 5 records:", df.head())
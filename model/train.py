from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import javalang


# Tokenization function for Java files
def tokenize(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    # Tokenize using javalang's tokenizer
    tokens = list(javalang.tokenizer.tokenize(code))
    return [token.value for token in tokens]


# Calculate token overlap (Jaccard similarity) between two files
def calculate_token_overlap(file1, file2):
    tokens1 = set(tokenize(file1))
    tokens2 = set(tokenize(file2))
    overlap = len(tokens1 & tokens2) / len(tokens1 | tokens2)
    return overlap


# Calculate structural (AST) similarity
def calculate_ast_similarity(file1, file2):
    def parse_ast(file_path):
        with open(file_path, 'r') as file:
            code = file.read()
        # Parse the AST
        tree = javalang.parse.parse(code)
        return tree

    tree1 = parse_ast(file1)
    tree2 = parse_ast(file2)

    # Compare structure by counting nodes or types in AST
    nodes1 = [node.__class__.__name__ for _, node in tree1]
    nodes2 = [node.__class__.__name__ for _, node in tree2]
    vectorizer = CountVectorizer().fit(nodes1 + nodes2)

    vec1 = vectorizer.transform([' '.join(nodes1)]).toarray()
    vec2 = vectorizer.transform([' '.join(nodes2)]).toarray()

    return cosine_similarity(vec1, vec2)[0][0]


# Calculate semantic similarity using embeddings
# (placeholder example with cosine similarity)
def calculate_semantic_similarity(file1, file2):
    with open(file1, 'r') as f:
        text1 = f.read()
    with open(file2, 'r') as f:
        text2 = f.read()

    # Vectorize code as raw text; replace this with code embeddings for
    # more accuracy
    vectorizer = CountVectorizer().fit([text1, text2])
    vec1 = vectorizer.transform([text1]).toarray()
    vec2 = vectorizer.transform([text2]).toarray()

    return cosine_similarity(vec1, vec2)[0][0]


def load_dataset(path):
    full_paths = list(path.iterdir())
    return full_paths


def load_original_file(path):
    original_file_path_directory = str(path) + "/original"
    original_file_path_file = Path(original_file_path_directory)
    original_file = list(original_file_path_file.iterdir())

    return original_file[0]


def load_plagiarized_files(path):
    plagiarized_path = str(path) + "/plagiarized"
    plagiarized_path = Path(plagiarized_path)
    plagiarized_L_directories = list(plagiarized_path.iterdir())
    plagiarized_files_paths = []
    for path in plagiarized_L_directories:
        plagiarized_files_paths = plagiarized_files_paths + list(path.iterdir())

    plagiarized_files = []
    for path in plagiarized_files_paths:
        plagiarized_files = plagiarized_files + list(path.iterdir())

    return plagiarized_files


def load_non_plagiarized_files(path):
    non_plagiarized_path = str(path) + "/non-plagiarized"
    non_plagiarized_path = Path(non_plagiarized_path)

    non_plagiarized_inner = list(non_plagiarized_path.iterdir())

    non_plagiarized_files = []
    for path in non_plagiarized_inner:
        non_plagiarized_files = non_plagiarized_files + list(path.iterdir())

    return non_plagiarized_files


# Prepare features and labels
features = []
labels = []

dataset = Path("/home/stormblessed/parser-plagio/data/IR-Plag-Dataset/")
cases = load_dataset(dataset)

# Loop through your dataset, computing similarity features and labels
for case in cases:  # This assumes you have a list or structure with paths to your cases
    original_file = load_original_file(case)  # Load path to the original file in each case
    plagiarized_files = load_plagiarized_files(case)
    non_plagiarized_files = load_non_plagiarized_files(case)
    for file in plagiarized_files + non_plagiarized_files:  # Paths to plagiarized and non-plagiarized files
        token_overlap = calculate_token_overlap(original_file, file)
        ast_similarity = calculate_ast_similarity(original_file, file)
        semantic_similarity = calculate_semantic_similarity(original_file, file)

        # Append features and labels
        features.append([token_overlap, ast_similarity, semantic_similarity])
        labels.append(get_label(case, file))  # Replace with actual plagiarism percentage or binary label

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Split and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

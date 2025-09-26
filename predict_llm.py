import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pandas as pd
import torch
import numpy as np
from argparse import Namespace
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self, args):
        self.args = args
        self.le = None
        self.isPreprocess = False
        self.correct_lookup = None

    def load_data(self):
        self.train_df = pd.read_csv(self.args.train_path)
        self.test_df = pd.read_csv(self.args.test_path)
        if self.args.use_extra_data:
            # This part is conditional and I don't have the extra data, so I'll skip it.
            pass

    def get_num_classes(self):
        if not self.isPreprocess:
            raise ValueError("Please preprocess first")
        return self.train_df['label'].nunique()

    def get_label_encoder(self):
        if self.le is None:
            raise ValueError("LabelEncoder not initialized. Please run preprocess first.")
        return self.le

    @staticmethod
    def format_input(row):
        correct_text = "Yes" if row['IsCorrect'] else "No"
        return (
            f"Question: {row['QuestionText']}\n"
            f"Answer: {row['MC_Answer']}\n"
            f"Correct? {correct_text}\n"
            f"Student Explanation: {row['StudentExplanation']}\n"
        )

    def preprocess(self):
        self.load_data()
        self.train_df['Misconception'] = self.train_df['Misconception'].fillna('NA')
        self.train_df['target'] = self.train_df['Category'] + ':' + self.train_df['Misconception']

        correct_samples = self.train_df[self.train_df['Category'].str.startswith('True', na=False)].copy()
        correct_samples['count'] = correct_samples.groupby(['QuestionId', 'MC_Answer'])['MC_Answer'].transform('count')
        most_popular_correct = correct_samples.sort_values('count', ascending=False).drop_duplicates(['QuestionId'])
        self.correct_lookup = most_popular_correct[['QuestionId', 'MC_Answer']].copy()
        self.correct_lookup['IsCorrect_flag'] = True

        self.train_df = self.train_df.merge(self.correct_lookup, on=['QuestionId', 'MC_Answer'], how='left')
        self.train_df['IsCorrect'] = self.train_df['IsCorrect_flag'].notna()
        self.train_df = self.train_df.drop(columns=['IsCorrect_flag'])

        self.le = LabelEncoder()
        self.train_df['label'] = self.le.fit_transform(self.train_df['target'])
        self.train_df['text'] = self.train_df.apply(self.format_input, axis=1)

        self.isPreprocess = True
        print("Data preprocessing complete.")
        return self.train_df

    def inference_processor(self):
        if not self.isPreprocess:
            raise ValueError("Have you run the train preprocessing? Please run preprocess first")
        self.test_df = self.test_df.merge(self.correct_lookup, on=['QuestionId', 'MC_Answer'], how='left')
        self.test_df['IsCorrect'] = self.test_df['IsCorrect_flag'].notna()
        self.test_df = self.test_df.drop(columns=['IsCorrect_flag'])
        self.test_df['text'] = self.test_df.apply(self.format_input, axis=1)
        print("Inference data processing complete.")
        return self.test_df

# --- Main execution block ---
if __name__ == '__main__':
    args = Namespace(
        train_path='train.csv',
        test_path='test.csv',
        use_extra_data=False,
        extra_data_path='datas.csv', # This file doesn't exist, so use_extra_data is False
        model_dir="model",
        inference_model_dir="model/inference_model",
        mode='inference',
        model_name="model" # Using the placeholder directory
    )

    DP = DataProcessor(args)
    train_data = DP.preprocess()
    test_data = DP.inference_processor()

    print("\nPreprocessed Training Data Head:")
    print(train_data.head())
    print("\nPreprocessed Test Data Head:")
    print(test_data.head())

    # --- MOCK INFERENCE & SUBMISSION GENERATION ---
    # NOTE: The model loading and real inference steps are skipped due to environmental
    # constraints (inability to download large model files).
    # Instead, we generate mock predictions to demonstrate the final step.
    print("\n--- SKIPPING REAL INFERENCE ---")
    print("Generating mock predictions to simulate model output.")

    num_test_samples = len(test_data)
    num_classes = DP.get_num_classes()
    # Simulate probabilities output by the model
    mock_probs = np.random.rand(num_test_samples, num_classes)
    print(f"Generated mock probabilities of shape: {mock_probs.shape}")

    # --- Generate submission.csv ---
    print("\nGenerating submission file...")
    top3_indices = np.argsort(-mock_probs, axis=1)[:, :3]

    le = DP.get_label_encoder()

    # Ensure all indices are within the range of the label encoder's classes
    valid_indices = np.where(top3_indices < len(le.classes_))

    # Create a placeholder for decoded labels
    decoded_labels_flat = np.full(top3_indices.flatten().shape, 'NA:NA', dtype=object)

    # Decode only the valid indices
    flat_top3 = top3_indices.flatten()
    decoded_labels_flat = le.inverse_transform(flat_top3)

    top3_labels = decoded_labels_flat.reshape(top3_indices.shape)
    joined_preds = [" ".join(row) for row in top3_labels]

    submission_df = pd.DataFrame({
        "row_id": test_data.row_id.values,
        "Category:Misconception": joined_preds
    })
    submission_df.to_csv("submission.csv", index=False)

    print("Submission file created successfully: submission.csv")
    print("\nSubmission file head:")
    print(submission_df.head())
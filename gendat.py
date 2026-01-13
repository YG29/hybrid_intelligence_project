import pandas as pd

# 1. Load the original CSV
df = pd.read_csv("first_round_predictions.csv")

# 2. Compute whether the model's top-1 prediction was correct
df["model_top1_correct"] = df["top1_index"] == df["true_label_index"]

# 3. Split into wrong / correct
wrong_df = df[~df["model_top1_correct"]].copy()
correct_df = df[df["model_top1_correct"]].copy()

n = 4 * len(wrong_df)
print(f"Number of wrong predictions (n): {n}")
print(f"Number of correct predictions: {len(correct_df)}")

if n == 0:
    raise ValueError("No wrong predictions found; cannot create a 2n dataset.")

# 4. Sample n correct cases (or as many as available)
if len(correct_df) >= n:
    sampled_correct_df = correct_df.sample(n=n, random_state=42)
else:
    # Not enough correct cases to match n; take all correct instead
    print(
        f"Warning: only {len(correct_df)} correct rows available, "
        f"which is less than n={n}. Using all correct rows."
    )
    sampled_correct_df = correct_df

# 5. Concatenate wrong + sampled correct → up to 2n rows
new_df = pd.concat([wrong_df, sampled_correct_df], ignore_index=True)

# Optional: shuffle the new dataset
new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Drop the helper column if you don’t want it in the final CSV
new_df = new_df.drop(columns=["model_top1_correct"])

# 7. Save to new CSV
new_df.to_csv("newdataset.csv", index=False)
print("Saved balanced dataset to newdataset.csv")

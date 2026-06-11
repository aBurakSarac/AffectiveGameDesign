import argparse
import csv
import os
import sys

def process_csv(input_csv):
    if not os.path.exists(input_csv):
        print(f"Error: Could not find {input_csv}")
        return

    output_csv = input_csv.replace(".csv", "_compact.csv")
    if output_csv == input_csv:
        output_csv = input_csv.replace(".csv", "") + "_compact.csv"

    print(f"Reading: {input_csv}")
    print(f"Writing: {output_csv}")

    kept_rows = 0
    total_rows = 0

    with open(input_csv, "r", encoding="utf-8") as infile, \
         open(output_csv, "w", newline="", encoding="utf-8") as outfile:
         
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        try:
            header = next(reader)
            writer.writerow(header)
        except StopIteration:
            print("CSV is empty.")
            return

        # Build index map to find columns regardless of column order
        idx = {col_name: i for i, col_name in enumerate(header)}

        for row in reader:
            if not row:
                continue
            
            total_rows += 1

            try:
                mp_face_detected = int(row[idx.get("mp_face_detected", -1)]) if "mp_face_detected" in idx else 1
                hs_face_detected = int(row[idx.get("hs_face_detected", -1)]) if "hs_face_detected" in idx else 1
                event_status = row[idx.get("event_status", -1)] if "event_status" in idx else "IDLE"
                startle_score = float(row[idx.get("mp_startle_score", -1)]) if "mp_startle_score" in idx else 0.0
                tension = float(row[idx.get("mp_tension", -1)]) if "mp_tension" in idx else 0.0
                composite_fear = float(row[idx.get("composite_fear", -1)]) if "composite_fear" in idx else 0.0
                hs_neutral_val = float(row[idx.get("hs_neutral", -1)]) if "hs_neutral" in idx else 1.0
            except (ValueError, IndexError):
                # If row parsing fails, default to keeping it to be safe
                writer.writerow(row)
                kept_rows += 1
                continue

            # LLM Optimization Filtering Logic
            is_boring = True
            if not (mp_face_detected or hs_face_detected):
                is_boring = True
            elif event_status != "IDLE":
                is_boring = False
            elif startle_score > 0.5:
                is_boring = False
            elif tension > 0.25:
                is_boring = False
            elif composite_fear > 0.1:
                is_boring = False
            elif hs_neutral_val < 0.70:
                is_boring = False

            if not is_boring:
                writer.writerow(row)
                kept_rows += 1

    print(f"Done! Kept {kept_rows} / {total_rows} frames ({kept_rows/max(total_rows,1)*100:.1f}%).")
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an LLM-optimized compact CSV from a full session CSV.")
    parser.add_argument("input_csv", type=str, help="Path to the full session CSV.")
    args = parser.parse_args()

    process_csv(args.input_csv)

# Submission & Testing Plan

This plan guides you through verifying the final pipeline results and preparing your deliverables for the technical test.

## Phase 1: Process the Full Match
Since the previous tracking run was stopped at 50 minutes, you need to re-run it and let it finish the entire 94-minute video.
1. Open a terminal and make sure you are in the virtual environment.
2. Run the tracking script:
   ```bash
   .venv/bin/python pipeline/01_track.py --video data/raw/match.mp4
   ```
3. **Wait** until the progress bar reaches 100% and tracking is complete (it will take roughly 1 hour and 40 minutes on your machine at ~4.6 frames/s).
4. Do not stop it early! Once finished, it will output: `Done. X detections across Y tracks`.

## Phase 2: Run Stats & Fatigue
Once the full tracking is completed, run the fast scripts to compute the full 90-minute distances and risk scores.
1. Run the stats script:
   ```bash
   .venv/bin/python pipeline/02_stats.py
   ```
2. **Check the sanity output in the terminal.** 
   - Look for the `Median distance` line.
   - Because you processed the full match, it should land nicely between **6 km and 12 km** (since we already calibrated it to ~4.7km for a single half). Proceed to Step 3.
3. Run the fatigue script:
   ```bash
   .venv/bin/python pipeline/03_fatigue.py
   ```
   - You should see a list of players with `LOW`, `MEDIUM`, and `HIGH` scores, plus some `INSUFFICIENT` fragments.

## Phase 3: Verify the Dashboard
1. Launch the dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```
2. Open your browser to the local URL (usually `http://localhost:8501`).
3. **Verify Team Overview:**
   - Check the KPI boxes at the top (Match Duration, Avg Distance, etc.).
   - Verify the donut chart loads.
   - Verify the bar chart (Distance per 15-min block) loads.
4. **Verify Player Detail:**
   - Select a scored player (not INSUFFICIENT) from the sidebar.
   - Verify the Heatmap renders correctly (no crashes!).
   - Verify the Fatigue & Risk Breakdown bar chart appears next to the heatmap.
   - Select an `INSUFFICIENT` player and verify the yellow warning banner appears.

## Phase 4: Prepare the Deliverables
1. **Record the Demo Video (3-5 minutes):**
   - Use QuickTime Player (File > New Screen Recording).
   - Briefly show your code structure (`pipeline/` and `dashboard/`).
   - Run `streamlit run dashboard/app.py` on camera.
   - Walk through the Team Overview page.
   - Walk through a specific Player Detail page, highlighting the Heatmap and Fatigue indicators.
   - Mention the "Naive ID Stitching" you added to handle halftime tracking losses.
   - Upload this video to YouTube (Unlisted) or Google Drive, or keep the file ready to send.
2. **Review the README:**
   - Open `README.md` and ensure it accurately reflects your work (the updated version handles the calibration, ID switches, and INSUFFICIENT logic).
   - Ensure the required questions from the test prompt are answered in the README.

## Phase 5: GitHub Submission
1. Check your git status:
   ```bash
   git status
   ```
2. Ensure your `match.mp4` and `data/processed/` files are **NOT** staged (they should be ignored by `.gitignore`).
3. Commit your code:
   ```bash
   git add .
   git commit -m "Final submission: YOLOv8m tracking, stitched ID fatigue scoring, and Streamlit dashboard"
   ```
4. Push to your GitHub repository:
   ```bash
   git branch -M main
   git push -u origin main
   ```

## Phase 6: Final Email
Send the email to the supervisor (`ao@estepshealth.com`) containing:
1. The link to your public/private GitHub repository.
2. The link to your Demo Video.
3. A brief thank you note.
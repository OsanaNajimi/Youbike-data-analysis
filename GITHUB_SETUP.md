# GitHub Actions Setup Guide

This will automatically collect bike data every 10 minutes using GitHub's servers (free!).

## 📋 Prerequisites

- GitHub account (free)
- Git installed on your computer

## 🚀 Setup Steps (5 minutes)

### Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the **"+"** in the top right → **"New repository"**
3. Repository name: `youbike-data` (or whatever you want)
4. Make it **Public** or **Private** (both work)
5. **Do NOT** initialize with README
6. Click **"Create repository"**

### Step 2: Push Your Code to GitHub

Open your terminal in this folder and run:

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit with data collection setup"

# Add your GitHub repository (replace USERNAME and REPO_NAME)
git remote add origin https://github.com/USERNAME/REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Replace:**
- `USERNAME` with your GitHub username
- `REPO_NAME` with your repository name (e.g., `youbike-data`)

**Example:**
```bash
git remote add origin https://github.com/jimmy123/youbike-data.git
```

### Step 3: Enable GitHub Actions

1. Go to your repository on GitHub
2. Click the **"Actions"** tab
3. You should see "Collect YouBike Data" workflow
4. Click **"Enable workflow"** if prompted

### Step 4: Trigger First Collection (Test)

1. In the **Actions** tab, click on **"Collect YouBike Data"**
2. Click **"Run workflow"** dropdown on the right
3. Click the green **"Run workflow"** button
4. Wait ~30 seconds and refresh the page
5. You should see a green checkmark ✓

### Step 5: Verify It's Working

1. Go to the **"Code"** tab
2. Open the `bike_data/` folder
3. You should see a new CSV file!
4. Click on "commits" to see the auto-commit from GitHub Actions

## ⏰ How It Works

- **Every 10 minutes**, GitHub Actions will:
  1. Run `github_collect.py`
  2. Collect current bike data (one snapshot)
  3. Save to `bike_data/` folder
  4. Automatically commit and push to your repo

- **After 24 hours**, you'll have **144 data points** (24 hours × 6 per hour)
- **After 48 hours**, you'll have **288 data points**

## 📥 Download Your Data

After collecting for a day or two:

```bash
# Pull the latest data from GitHub
git pull

# Run analysis
python analyze_regression.py
```

## 🛑 How to Stop Collection

1. Go to your GitHub repository
2. Click **"Actions"** tab
3. Click **"Collect YouBike Data"**
4. Click **"..."** (three dots) in the top right
5. Click **"Disable workflow"**

Or just delete the repository when done!

## 💰 Cost

**FREE!** GitHub gives you 2,000 free Action minutes per month. Each collection takes ~30 seconds, so:
- 10 min intervals = 6 runs/hour = 144 runs/day
- 144 runs × 30 seconds = 4,320 seconds = 72 minutes/day
- You can run this for **27+ days** on the free tier!

## 🔧 Troubleshooting

### "workflow not found"
- Make sure `.github/workflows/collect_data.yml` was uploaded
- Check file is in correct folder structure

### "Permission denied"
1. Go to repository **Settings**
2. Click **Actions** → **General** (left sidebar)
3. Scroll to **Workflow permissions**
4. Select **"Read and write permissions"**
5. Click **Save**

### Data not showing up
- Check the Actions tab for error messages
- Click on a failed run to see logs
- Make sure `bike_data/` folder exists in repo

## 📊 View Collection Progress

Check how many data points you have:
1. Go to your GitHub repo
2. Open `bike_data/` folder
3. Count the CSV files (or check last commit time)

## ✅ Next Steps

Once you have 24+ hours of data:
1. Pull from GitHub: `git pull`
2. Run analysis: `python analyze_regression.py`
3. View results in `regression_results/` folder

Good luck! 🚀


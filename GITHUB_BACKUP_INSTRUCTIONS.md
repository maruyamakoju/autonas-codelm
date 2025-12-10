# GitHub Backup Instructions

**Status**: Ready to push. Follow these steps to backup to GitHub.

---

## Step 1: Create GitHub Repository (Web Browser)

1. Go to https://github.com/new
2. Repository settings:
   - **Name**: `autonas-codelm` (or any name you prefer)
   - **Description**: Multi-objective NAS for code language models with mode collapse study
   - **Visibility**: Private or Public (your choice)
   - **Initialize**: ❌ DO NOT check "Add README", "Add .gitignore", or "Choose a license"
     - The repository MUST be completely empty

3. Click "Create repository"
4. Copy the repository URL from the page (you'll need this in Step 2)
   - SSH: `git@github.com:USERNAME/autonas-codelm.git`
   - HTTPS: `https://github.com/USERNAME/autonas-codelm.git`

---

## Step 2: Push to GitHub (Command Line)

Open terminal in project directory and run:

```bash
# Navigate to project directory (if not already there)
cd C:/Users/07013/Desktop/1205muzi5090

# Add remote (replace URL with your repository URL from Step 1)
git remote add origin git@github.com:USERNAME/autonas-codelm.git
# Or if using HTTPS:
# git remote add origin https://github.com/USERNAME/autonas-codelm.git

# Push all commits and tags
git push -u origin master
git push origin v1.0-strongreg

# Verify
git remote -v
```

---

## Step 3: Verify on GitHub

Go to your repository URL in browser and verify:
- ✅ All files are present
- ✅ Tag `v1.0-strongreg` appears in releases/tags
- ✅ Latest commit: "Complete v1 closure: instruction tuning experiments + v2 planning"

---

## What's Being Backed Up

**v1 artifacts**:
- Training code, models, evaluation scripts
- Documentation: NOTES_V1_DONE.md, STRONGREG_SUMMARY.md
- Git tag: v1.0-strongreg

**Instruction tuning experiments**:
- Failed experiments documented in INSTRUCTION_TUNING_FAILURE_REPORT.md
- Training scripts and datasets (49 samples)

**v2 planning**:
- V2_PLAN.md: Options for scaling up quality
- DATA_EXPANSION_GUIDE.md: 500-1000 sample expansion plan

**Note**: Model checkpoints (`.pt` files) and large datasets are excluded via `.gitignore`. Only code, configs, and documentation are backed up.

---

## After Backup Complete

Add this line to `NOTES_V1_DONE.md`:

```markdown
**GitHub Repository**: https://github.com/USERNAME/autonas-codelm
```

---

**Status after completion**: ✅ v1 project fully backed up and safe to archive or continue.

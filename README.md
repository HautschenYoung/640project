# 640project

## Install DUSt3R as a Git submodule (development / dependency)

This project depends on the external DUSt3R repository. The recommended workflow is to add DUSt3R as a git submodule and install it in editable mode so changes in the submodule are available to your environment.

1. Add the submodule (run from your project root)
```bash
git submodule add https://github.com/naver/dust3r pj/dust3r
git submodule update --init --recursive
git add .gitmodules pj/dust3r
git commit -m "Add dust3r as git submodule"
```

2. Clone / update for collaborators
```bash
# clone with submodules
git clone --recurse-submodules <your-repo-url>

# if already cloned
git submodule update --init --recursive
```

3. Install DUSt3R dependencies and make the submodule available to Python
```bash
# install dependencies used by dust3r (see dust3r/requirements.txt for exact list)
pip install -r pj/dust3r/requirements.txt

# install dust3r in editable mode so imports work and local updates are reflected
pip install -e pj/dust3r
```

4. Verify import
```bash
python -c "import dust3r; print('dust3r import OK')"
```
# How to build the doc site locally

```bash
pip install -r requirements_doc.txt
rm -rf _build/ && sphinx-build -b html . _build/html/
cd _build/html/ && python -m http.server
```

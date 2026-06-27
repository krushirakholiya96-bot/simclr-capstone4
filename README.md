# SimCLR Contrastive Representation Learner

![CI/CD](https://github.com/krushirakholiya96-bot/simclr-capstone4/actions/workflows/ci.yml/badge.svg)

## Live Demo
- API: https://your-app.onrender.com
- Dashboard: https://your-app.streamlit.app

## Project Summary
Built end-to-end SimCLR Contrastive Self-Supervised Learning system on CIFAR-10 
using ResNet50 achieving 83%+ accuracy. Integrated Groq Llama 3 for Generative AI 
explainability and an 8-step Agentic AI pipeline for autonomous image analysis. 
Deployed via FastAPI + Streamlit with Docker + GitHub Actions CI/CD — 100% free stack.

## Tech Stack
- ML: PyTorch + ResNet50 + SimCLR
- Generative AI: Groq Llama 3
- Agentic AI: 8-step autonomous pipeline
- Backend: FastAPI + SQLite
- Frontend: Streamlit Dark Dashboard
- DevOps: Docker + GitHub Actions
- Deployment: Render.com + Streamlit Cloud

## Results
| Method | Accuracy |
|--------|----------|
| Linear Probe | 73% |
| SimCLR Fine-tuned | 83%+ |
| Supervised Baseline | 87% |

## Installation
```bash
git clone https://github.com/krushirakholiya96-bot/simclr-capstone4.git
cd simclr-capstone4
pip install -r requirements.txt
```

## Run
```bash
# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Start Dashboard
streamlit run dashboard/app.py
```

## Docker
```bash
docker-compose up --build
```
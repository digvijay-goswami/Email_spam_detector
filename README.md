# Email Spam Detection

A beginner-friendly, deployable minor project that classifies an email or short message as **ham** (normal) or **spam**. It uses a Multinomial Naive Bayes model written in plain Python - no paid API, hardware, or machine-learning library is required.

## Why this is a good first project

- The problem is familiar and useful: spam wastes time and can lead to fraud.
- You can explain every part of the model in a viva.
- It has a visible web interface, a saved model, tests, evaluation metrics, and a deployment configuration.
- You can improve it later by training on a bigger labelled dataset.

## Project scope for your report

**Title:** Email Spam Detection System Using Multinomial Naive Bayes

**Problem statement:** Users receive unwanted and deceptive messages. The system should automatically classify a message as spam or ham, helping a user decide when to be cautious.

**Inputs:** The text of an email/message.

**Output:** A predicted label (`spam` or `ham`) and a confidence score.

**Algorithm:** Multinomial Naive Bayes with Laplace smoothing.

**Current dataset:** `data/sample_messages.csv` contains 95 small labelled examples to make the project work immediately. It is deliberately small, so its score is only a demonstration. For your final submission, train with a larger public dataset and state the source, licence, number of messages, class balance, and cleaning steps in your report. A suitable option is the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms), which has 5,574 labelled messages and is released under CC BY 4.0.

## How it works

1. Each labelled message is split into lowercase words (tokens).
2. During training, the program counts how often every word occurs in ham and spam messages.
3. For a new message, it combines the probability of its words under each class. Laplace smoothing gives unseen words a small non-zero probability.
4. The class with the higher probability is shown in the website.
5. The training program reserves 20% of each class for a test, prints accuracy/precision/recall/F1, then retrains the final saved model with all available labelled data.

The model is a **bag-of-words** model: it knows which words appeared, not word order or the sender identity. This keeps it simple and explainable, but it also means the confidence is an estimate, not proof that a message is safe.

## Run it locally

You only need Python 3.10 or newer.

```bash
cd email-spam-detector
python train.py
python -m unittest discover -s tests
python app.py
```

Open `http://localhost:8080` in your browser. Stop the server with `Ctrl+C`.

On Windows, if `python` is not recognised, try `py` in place of `python`.

## Retrain it with a larger dataset

1. Save a CSV in `data/` with columns named `label,message`. Use `ham` and `spam` as labels. The common `v1,v2` columns are also supported. The native tab-separated `SMSSpamCollection` file from UCI can be used directly too.
2. Run the command below, changing the filename if needed:

```bash
python train.py --data data/your_dataset.csv

# Or, after downloading and extracting UCI's file:
python train.py --data data/SMSSpamCollection
```

3. Read the printed metrics and include them in your report. Start with precision, recall, and F1 score - accuracy alone can look good when there are far more ham messages than spam messages.
4. Restart `python app.py` so it loads the new `model/model.json`.

Do not put private emails into a dataset or upload them to a public repository.

## Deploy with Docker

The included `Dockerfile` makes the app portable to any Docker-compatible host.

```bash
docker build -t email-spam-detector .
docker run --rm -p 8080:8080 -e PORT=8080 email-spam-detector
```

Then open `http://localhost:8080`.

### Deploy to Render

1. Create a new GitHub repository and upload this whole `email-spam-detector` folder.
2. In Render, create a new **Web Service**, connect the repository and choose the Docker runtime. Render will use the included `Dockerfile`.
3. Leave the service port to the platform. The app reads its port from the `PORT` environment variable.
4. Deploy. The `/health` endpoint is included for the platform's health check.
5. Open the URL Render gives you and test one ham and one spam example.

The `render.yaml` file supplies the service type and health-check path if you use Render's Blueprint flow.

## Deploy with Vercel

This project also works on Vercel. Unlike Docker hosting, Vercel runs a small Python function every time someone visits the site. The `api/` folder and `vercel.json` already adapt the website for that style of hosting.

1. Upload this project's files to a GitHub repository, with `vercel.json` at the repository root.
2. In Vercel, select **Add New > Project**, import that GitHub repository, and keep the root directory as the project root.
3. Choose **Other** if Vercel asks for a framework preset, then click **Deploy**. No build or start command is needed.
4. Vercel gives you a public `https://...vercel.app` address. Open it and try the normal and spam demonstration messages above.

Vercel uses the files in `api/index.py` and `api/health.py`; do not run those files yourself. Continue to use `python app.py` for local testing.

## Suggested demonstration examples

- Normal: `Could we meet at the library tomorrow at 3 pm?`
- Spam: `You won a free cash prize! Click now to claim it.`

## Viva-ready explanation

If asked why Naive Bayes was selected: *It is fast, works well as a baseline for text classification, needs little computing power, and is easy to interpret. I used Laplace smoothing so a word not seen in training does not make the entire probability zero.*

If asked about limitations: *The model can be fooled by new wording or short messages, and it does not inspect links, attachments, sender reputation, or grammar. A production system should use more data, stronger models, and human/security checks.*

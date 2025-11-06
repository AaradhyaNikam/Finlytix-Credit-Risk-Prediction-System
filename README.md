# Finlytix: AI Credit Risk Intelligence System 💰

A modern, explainable AI system for credit risk assessment with an interactive dashboard and chatbot interface.

## Features 🌟

- **Interactive Dashboard**: Clean, modern UI with real-time credit risk predictions
- **Explainable AI**: Uses SHAP values to provide transparent decision explanations
- **Chatbot Interface**: Natural language interaction for credit risk queries
- **Risk Visualization**: Clear metrics and visual representations of risk factors
- **Custom Themed UI**: Professional dark theme with financial aesthetics

## System Components 📦

1. **Dashboard Mode**
   - Real-time credit risk prediction
   - Risk level categorization (High/Medium/Low)
   - Interactive data visualization
   - SHAP-based feature importance analysis
   - Customer data display

2. **Chatbot Mode**
   - Natural language query processing
   - Structured risk assessment responses
   - Historical conversation tracking
   - Easy-to-understand explanations

## Technical Stack 🛠️

- **Frontend**: Streamlit with custom CSS theming
- **Backend**: Python with XGBoost model
- **ML Components**:
  - XGBoost Classifier for prediction
  - SHAP for model explanability
  - Scikit-learn for data preprocessing
  - Custom preprocessing pipeline

## Project Structure 📁

```
├── final_app.py          # Main application (Dashboard + Chatbot)
├── train_model.py        # Initial model training script
├── train_final.py        # Production model training script
├── sampleEntry.csv       # Sample data entry format
├── testing-1.csv         # Test dataset
├── training.csv          # Training dataset
├── final_model.pkl       # Trained production model
├── final_pipeline.pkl    # Production preprocessing pipeline
```

## Installation & Setup 🚀

1. Clone the repository:
```bash
git clone https://github.com/AaradhyaNikam/Finlytix-project.git
cd Finlytix-project
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run final_app.py
```

## Usage Guide 📖

### Dashboard Mode
1. Select "📊 Dashboard" from the sidebar
2. Enter a customer ID in the input field
3. View the risk assessment results and explanations
4. Explore SHAP insights and customer data

### Chatbot Mode
1. Select "💬 Chatbot" from the sidebar
2. Type queries in the format: "check id <number>"
3. View structured responses with risk assessments
4. Review historical conversation in the chat interface

## Model Details 🤖

- **Algorithm**: XGBoost Classifier
- **Features**: Numerical and categorical data processing
- **Evaluation Metric**: ROC-AUC
- **Explainability**: SHAP (SHapley Additive exPlanations)

## Contributing 🤝

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss the proposed changes.

---

Created by Aaradhya Nikam | Finlytix Project
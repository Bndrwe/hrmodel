"""Train and adjust the model based on actual results using AI."""
import json
import os
from pathlib import Path
from datetime import datetime, date, timedelta
import requests

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def load_predictions(date_str):
    """Load predictions for a specific date."""
    hist_file = Path(f"data/history/predictions_{date_str}.json")
    if hist_file.exists():
        with open(hist_file) as f:
            return json.load(f)
    return None

def load_results(date_str):
    """Load actual results for a specific date."""
    results_file = Path(f"data/results/results_{date_str}.json")
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None

def compare_predictions_to_results(predictions, results):
    """Compare predicted vs actual HRs."""
    if not predictions or not results:
        return []
    
    comparisons = []
    results_dict = {r['player_name']: r for r in results.get('results', [])}
    
    for pred in predictions.get('predictions', [])[:50]:  # Top 50
        player_name = pred['name']
        predicted_prob = pred['game_prob']
        
        if player_name in results_dict:
            actual_hrs = results_dict[player_name]['home_runs']
            hit_hr = actual_hrs > 0
            
            comparisons.append({
                'player': player_name,
                'predicted_prob': predicted_prob,
                'actual_hrs': actual_hrs,
                'hit_hr': hit_hr,
                'grade': pred.get('grade', 'N/A'),
                'factors': pred.get('factors', {})
            })
    
    return comparisons

def analyze_with_ai(comparisons):
    """Use OpenRouter AI to analyze patterns and suggest improvements."""
    if not OPENROUTER_API_KEY:
        print("No OpenRouter API key - skipping AI analysis")
        return None
    
    # Prepare summary statistics
    total = len(comparisons)
    hits = sum(1 for c in comparisons if c['hit_hr'])
    accuracy = (hits / total * 100) if total > 0 else 0
    
    # Group by grade
    grade_stats = {}
    for c in comparisons:
        grade = c['grade']
        if grade not in grade_stats:
            grade_stats[grade] = {'total': 0, 'hits': 0}
        grade_stats[grade]['total'] += 1
        if c['hit_hr']:
            grade_stats[grade]['hits'] += 1
    
    prompt = f"""You are a data scientist analyzing a baseball home run prediction model.

Yesterday's Performance:
- Total predictions analyzed: {total}
- Actual home runs: {hits}
- Overall accuracy: {accuracy:.1f}%

Grade-level performance:
{json.dumps(grade_stats, indent=2)}

Sample predictions:
{json.dumps(comparisons[:10], indent=2)}

Based on this data:
1. Identify which factors or grades are overperforming or underperforming
2. Suggest specific weight adjustments (e.g., "increase park_factor by 5%", "decrease pitcher_adj by 3%")
3. Recommend any new factors to consider
4. Provide a confidence score (0-100) for these suggestions

Respond in JSON format with keys: analysis, weight_adjustments, new_factors, confidence_score"""
    
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        response.raise_for_status()
        
        ai_response = response.json()
        content = ai_response['choices'][0]['message']['content']
        
        # Try to parse JSON from response
        try:
            suggestions = json.loads(content)
        except:
            # If not valid JSON, wrap it
            suggestions = {"analysis": content, "confidence_score": 50}
        
        return suggestions
    
    except Exception as e:
        print(f"AI analysis error: {e}")
        return None

def save_training_log(date_str, comparisons, ai_suggestions):
    """Save training results to log."""
    log_file = Path("data/training_log.json")
    
    # Load existing log
    if log_file.exists():
        with open(log_file) as f:
            log = json.load(f)
    else:
        log = {"training_sessions": []}
    
    # Add new session
    session = {
        "date": date_str,
        "timestamp": datetime.now().isoformat(),
        "total_predictions": len(comparisons),
        "hits": sum(1 for c in comparisons if c['hit_hr']),
        "accuracy": (sum(1 for c in comparisons if c['hit_hr']) / len(comparisons) * 100) if comparisons else 0,
        "ai_suggestions": ai_suggestions
    }
    
    log["training_sessions"].append(session)
    
    # Keep only last 30 days
    log["training_sessions"] = log["training_sessions"][-30:]
    
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"Saved training log: {session['accuracy']:.1f}% accuracy")

def apply_weight_adjustments(ai_suggestions):
    """Apply AI-suggested weight adjustments to model weights file."""
    if not ai_suggestions or 'weight_adjustments' not in ai_suggestions:
        print("No weight adjustments to apply")
        return
    
    weights_file = Path("data/model_weights.json")
    
    # Load or create weights
    if weights_file.exists():
        with open(weights_file) as f:
            weights = json.load(f)
    else:
        weights = {"last_updated": None, "adjustments": []}
    
    # Add new adjustments
    weights["last_updated"] = datetime.now().isoformat()
    weights["adjustments"].append({
        "timestamp": datetime.now().isoformat(),
        "suggestions": ai_suggestions['weight_adjustments'],
        "confidence": ai_suggestions.get('confidence_score', 50)
    })
    
    # Keep only last 10 adjustments
    weights["adjustments"] = weights["adjustments"][-10:]
    
    with open(weights_file, 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"Applied weight adjustments (confidence: {ai_suggestions.get('confidence_score', 0)}%)")

def main():
    print("Starting model training...")
    
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Load data
    predictions = load_predictions(yesterday)
    results = load_results(yesterday)
    
    if not predictions or not results:
        print(f"Missing predictions or results for {yesterday}")
        return
    
    # Compare
    comparisons = compare_predictions_to_results(predictions, results)
    print(f"Compared {len(comparisons)} predictions to actual results")
    
    # Analyze with AI
    ai_suggestions = analyze_with_ai(comparisons)
    
    if ai_suggestions:
        print("AI Analysis completed:")
        print(json.dumps(ai_suggestions, indent=2))
        
        # Apply adjustments
        apply_weight_adjustments(ai_suggestions)
    
    # Save log
    save_training_log(yesterday, comparisons, ai_suggestions)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
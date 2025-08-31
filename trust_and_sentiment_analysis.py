import json
import datetime
import time
import re
from openai import OpenAI
from typing import Dict, List, Any, Optional

class OllamaTrustSentimentAnalyzer:
    def __init__(self, ollama_host="http://localhost:11434"):
        """Initialize Ollama-based Trust & Sentiment Analyzer"""
        self.ollama_host = ollama_host
        self.ollama_client = None
        self.current_model = None
        self.chat_history = []
        self.analysis_history = []
        
        # Single model configuration - only Qwen 3B
        self.models = {
            "1": {
                "name": "Qwen 2.5 3B Instruct",
                "model_id": "qwen2.5:3b",
                "family": "Qwen",
                "supports_thinking": False,
                "description": "Alibaba's powerful instruction model - comprehensive analysis"
            }
        }
        
        # Speed optimization settings
        self.disable_thinking = True  # Default to fast mode
        self.max_tokens = 1000  # Optimized for speed
        self.temperature = 0.1  # Consistent results
        
    def setup_ollama_client(self):
        """Setup OpenAI client pointing to Ollama"""
        try:
            self.ollama_client = OpenAI(
                base_url=f"{self.ollama_host}/v1",
                api_key="ollama"  # Required but unused
            )
            
            # Test connection
            models = self.ollama_client.models.list()
            print(f"✅ Connected to Ollama at {self.ollama_host}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Ollama: {e}")
            print("💡 Make sure Ollama is running: 'ollama serve'")
            return False
    
    def display_models(self):
        """Display available model with optimization info"""
        print("\n" + "="*70)
        print("🧠 OLLAMA TRUST & SENTIMENT ANALYSIS - QWEN MODEL")
        print("="*70)
        
        thinking_status = "🚀 FAST MODE" if self.disable_thinking else "🧠 THINKING MODE"
        print(f"Current Speed Setting: {thinking_status}")
        print(f"Max Tokens: {self.max_tokens} | Temperature: {self.temperature}")
        
        print("\n📋 Available Model:")
        model = self.models["1"]
        print(f"  1. {model['name']}")
        print(f"     Ollama ID: {model['model_id']}")
        print(f"     {model['description']}")
        print()
    
    def setup_model(self, model_choice: str = "1"):
        """Setup Qwen model"""
        if model_choice not in self.models:
            print("❌ Invalid model choice!")
            return False
            
        model_info = self.models[model_choice]
        print(f"\n🔄 Setting up {model_info['name']}...")
        
        # Test model availability
        try:
            test_response = self.ollama_client.chat.completions.create(
                model=model_info["model_id"],
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0.1
            )
            
            if test_response.choices:
                self.current_model = model_info
                print(f"✅ Successfully loaded {model_info['name']}")
                return True
            else:
                print(f"❌ No response from {model_info['name']}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing model: {e}")
            print(f"💡 Make sure model is installed: 'ollama pull {model_info['model_id']}'")
            return False
    
    def create_analysis_prompt(self, current_message: str, chat_history: List[Dict]) -> str:
        """Create optimized prompt for trust and sentiment analysis"""
        
        # Build concise chat history (max 3 messages for speed)
        history_text = ""
        if chat_history:
            recent_messages = chat_history[-3:]  # Last 3 messages
            for i, msg in enumerate(recent_messages, 1):
                history_text += f"Message {i}: {msg['content']}\n"
        
        # Optimized prompt - shorter for faster processing
        prompt = f"""You are an AI analyst for therapeutic conversations. Analyze trust and sentiment patterns.

Recent conversation context:
{history_text}

Current message: {current_message}

Analyze these aspects:
1. Trust trend: INCREASING/DECREASING/STABLE
2. Sentiment: IMPROVING/DECLINING/NEUTRAL

Consider:
- Openness and willingness to share
- Emotional tone and language
- Engagement level
- Resistance or cooperation

Return ONLY this JSON format:
{{
    "trust_trend": "INCREASING|DECREASING|STABLE",
    "trust_confidence": 0.8,
    "sentiment_trend": "IMPROVING|DECLINING|NEUTRAL",
    "sentiment_confidence": 0.8,
    "key_indicators": ["indicator1", "indicator2"],
    "analysis_summary": "Brief explanation",
    "timestamp": "{datetime.datetime.now().isoformat()}"
}}

JSON only, no other text."""
        
        return prompt
    
    def generate_analysis(self, prompt: str) -> str:
        """Generate analysis using Ollama with optimizations"""
        if not self.ollama_client or not self.current_model:
            raise Exception("Model not loaded")
        
        try:
            # Prepare messages
            messages = [
                {
                    "role": "system", 
                    "content": "You are a therapeutic conversation analyst. Respond only with valid JSON."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            # Generation parameters
            generation_params = {
                "model": self.current_model["model_id"],
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False
            }
            
            # Generate response
            response = self.ollama_client.chat.completions.create(**generation_params)
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Generation failed: {str(e)}")
    
    def parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate analysis response"""
        try:
            # Remove thinking tokens if they appear
            cleaned_response = self.clean_response(response)
            
            # Find JSON in response
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            json_str = json_match.group()
            analysis_result = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["trust_trend", "sentiment_trend", "trust_confidence", "sentiment_confidence"]
            for field in required_fields:
                if field not in analysis_result:
                    analysis_result[field] = "STABLE" if "trend" in field else 0.5
            
            return analysis_result
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            print("Response content:", response)
            return self.create_fallback_analysis(response)
        except Exception as e:
            print(f"⚠️ Analysis parsing error: {e}")
            return self.create_fallback_analysis(response)
    
    def clean_response(self, response: str) -> str:
        """Clean response by removing thinking tokens and extra content"""
        # Remove thinking blocks if present
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove common prefixes/suffixes
        cleaned = re.sub(r'^.*?(?=\{)', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\}.*?$', '}', cleaned, flags=re.DOTALL)
        
        return cleaned.strip()
    
    def create_fallback_analysis(self, raw_response: str) -> Dict[str, Any]:
        """Create fallback analysis when parsing fails"""
        return {
            "trust_trend": "STABLE",
            "trust_confidence": 0.5,
            "sentiment_trend": "NEUTRAL",
            "sentiment_confidence": 0.5,
            "key_indicators": ["parsing_error"],
            "analysis_summary": "Could not parse model response",
            "raw_response": raw_response[:200] + "..." if len(raw_response) > 200 else raw_response,
            "timestamp": datetime.datetime.now().isoformat(),
            "error": "response_parsing_failed"
        }
    
    def display_detailed_analysis(self, analysis: Dict[str, Any]) -> None:
        """Display detailed analysis results like in sample conversation"""
        if "error" not in analysis:
            print(f"🤝 Trust: {analysis.get('trust_trend')} ({analysis.get('trust_confidence', 0):.2f})")
            print(f"😊 Sentiment: {analysis.get('sentiment_trend')} ({analysis.get('sentiment_confidence', 0):.2f})")
            print(f"📝 {analysis.get('analysis_summary', 'No summary')}")
            print(f"⏱️  {analysis.get('processing_time_seconds', 0)}s")
            
            # Show key indicators if available
            if analysis.get("key_indicators"):
                indicators = ", ".join(analysis["key_indicators"])
                print(f"🔍 Key Indicators: {indicators}")
        else:
            print(f"❌ Error: {analysis['error']}")
    
    def perform_analysis(self, current_message: str) -> Dict[str, Any]:
        """Perform complete trust and sentiment analysis"""
        if not self.current_model:
            return {"error": "No model loaded", "timestamp": datetime.datetime.now().isoformat()}
        
        print(f"🔄 Analyzing with {self.current_model['name']}...")
        
        try:
            # Start timing
            start_time = time.time()
            
            # Create prompt
            prompt = self.create_analysis_prompt(current_message, self.chat_history)
            
            # Generate analysis
            response = self.generate_analysis(prompt)
            
            # Parse response
            analysis_result = self.parse_analysis_response(response)
            
            # Add performance metrics
            processing_time = time.time() - start_time
            analysis_result["processing_time_seconds"] = round(processing_time, 2)
            analysis_result["model_used"] = self.current_model["name"]
            
            # Store analysis
            self.analysis_history.append(analysis_result)
            
            print(f"✅ Analysis completed in {processing_time:.2f}s")
            return analysis_result
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "model_used": self.current_model["name"] if self.current_model else "unknown",
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.analysis_history.append(error_result)
            return error_result
    
    def add_user_message(self, message: str):
        """Add user message to chat history"""
        self.chat_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def adjust_speed_settings(self, max_tokens: int = None, temperature: float = None):
        """Adjust speed/quality settings"""
        if max_tokens is not None:
            self.max_tokens = max_tokens
            print(f"🔧 Max tokens set to: {max_tokens}")
        
        if temperature is not None:
            self.temperature = temperature
            print(f"🔧 Temperature set to: {temperature}")
    
    def display_analysis_summary(self):
        """Display comprehensive analysis summary"""
        print("\n" + "="*60)
        print("📊 ANALYSIS SUMMARY")
        print("="*60)
        
        if not self.analysis_history:
            print("No analyses performed yet.")
            return
        
        # Performance metrics
        total_time = 0
        successful_analyses = 0
        
        for i, analysis in enumerate(self.analysis_history, 1):
            has_error = "error" in analysis
            processing_time = analysis.get("processing_time_seconds", 0)
            model_used = analysis.get("model_used", "Unknown")
            
            print(f"\n📋 Analysis {i} - {model_used}")
            print(f"⏱️  Processing Time: {processing_time}s")
            
            if not has_error:
                successful_analyses += 1
                total_time += processing_time
                
                trust = analysis.get("trust_trend", "N/A")
                sentiment = analysis.get("sentiment_trend", "N/A")
                trust_conf = analysis.get("trust_confidence", 0)
                sentiment_conf = analysis.get("sentiment_confidence", 0)
                
                print(f"🤝 Trust: {trust} (confidence: {trust_conf:.2f})")
                print(f"😊 Sentiment: {sentiment} (confidence: {sentiment_conf:.2f})")
                print(f"📝 Summary: {analysis.get('analysis_summary', 'N/A')}")
                
                if analysis.get("key_indicators"):
                    indicators = ", ".join(analysis["key_indicators"])
                    print(f"🔍 Indicators: {indicators}")
            else:
                print(f"❌ Error: {analysis['error']}")
        
        # Overall performance summary
        if successful_analyses > 0:
            avg_time = total_time / successful_analyses
            print(f"\n📈 Performance Summary:")
            print(f"   Successful Analyses: {successful_analyses}/{len(self.analysis_history)}")
            print(f"   Average Processing Time: {avg_time:.2f}s")
            print(f"   Total Processing Time: {total_time:.2f}s")
            print(f"   Model Used: {self.current_model['name'] if self.current_model else 'None'}")
    
    def save_session_data(self, filename: str = "therapy_session_qwen.json"):
        """Save session data with model info"""
        session_data = {
            "session_info": {
                "model_used": self.current_model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "ollama_host": self.ollama_host
            },
            "chat_history": self.chat_history,
            "analysis_history": self.analysis_history,
            "session_end": datetime.datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        print(f"✅ Session saved to {filename}")

def create_sample_therapeutic_conversation():
    """Create sample conversation for testing"""
    return [
        "Hi, I'm feeling really anxious about my job interview tomorrow.",
        "I guess... I'm just worried I'll mess it up like last time.",
        "Well, you seem to understand. Maybe I can tell you more about what happened.",
        "You know what, talking about this is actually helping. I feel a bit better.",
        "I think I'm ready to try some of the techniques you suggested. Thank you for listening."
    ]

def main():
    """Main application loop"""
    print("🧠 OLLAMA TRUST & SENTIMENT ANALYZER - QWEN 2.5 3B")
    print("="*60)
    print("🚀 OpenAI Compatible • Speed Optimized • Local AI")
    print("="*60)
    
    # Initialize analyzer
    analyzer = OllamaTrustSentimentAnalyzer()
    
    # Setup Ollama connection
    print("\n🔌 Connecting to Ollama...")
    if not analyzer.setup_ollama_client():
        print("❌ Cannot connect to Ollama. Please start Ollama first.")
        return
    
    # Setup Qwen model automatically
    print("\n🎯 Setting up Qwen 2.5 3B model...")
    analyzer.display_models()
    
    print("\n🔧 Commands:")
    print("   'speed' - Adjust speed settings")
    print("   'q' - Quit")
    
    # Check if user wants to adjust settings or proceed
    choice = input("\nPress Enter to continue with default settings, 'speed' to adjust, or 'q' to quit: ").strip()
    
    if choice.lower() == 'q':
        return
    elif choice.lower() == 'speed':
        print("\n⚡ Speed Settings:")
        try:
            max_tokens = input(f"Max tokens (current: {analyzer.max_tokens}): ").strip()
            if max_tokens:
                analyzer.adjust_speed_settings(max_tokens=int(max_tokens))
            
            temperature = input(f"Temperature (current: {analyzer.temperature}): ").strip()
            if temperature:
                analyzer.adjust_speed_settings(temperature=float(temperature))
        except ValueError:
            print("⚠️ Invalid input, using defaults")
    
    # Setup model
    if not analyzer.setup_model("1"):
        print("❌ Failed to setup Qwen model")
        return
    
    # Demo options
    print("\n📋 Demo Options:")
    print("1. Sample therapeutic conversation")
    print("2. Custom message analysis")
    print("3. Interactive chat mode")
    
    demo_choice = input("\nChoose demo (1-3): ").strip()
    
    if demo_choice == "1":
        # Sample conversation
        sample_messages = create_sample_therapeutic_conversation()
        print(f"\n🎭 Running sample conversation analysis...")
        print(f"Model: {analyzer.current_model['name']}")
        
        for i, message in enumerate(sample_messages, 1):
            print(f"\n--- Message {i} ---")
            print(f"User: {message}")
            
            analyzer.add_user_message(message)
            analysis = analyzer.perform_analysis(message)
            
            # Display detailed results
            analyzer.display_detailed_analysis(analysis)
            
            input("Press Enter to continue...")
        
        analyzer.display_analysis_summary()
    
    elif demo_choice == "2":
        # Custom messages with detailed output
        print("\n✏️ Custom Message Analysis")
        print("Enter messages (type 'done' to finish):")
        
        while True:
            message = input("\nMessage: ").strip()
            if message.lower() == 'done':
                break
            
            analyzer.add_user_message(message)
            analysis = analyzer.perform_analysis(message)
            
            # Display detailed results like sample conversation
            analyzer.display_detailed_analysis(analysis)
        
        analyzer.display_analysis_summary()
    
    elif demo_choice == "3":
        # Interactive mode with detailed output
        print("\n💬 Interactive Chat Mode")
        print("Commands: 'quit', 'summary', 'save'")
        
        while True:
            message = input("\nYou: ").strip()
            
            if message.lower() == 'quit':
                break
            elif message.lower() == 'summary':
                analyzer.display_analysis_summary()
                continue
            elif message.lower() == 'save':
                analyzer.save_session_data()
                continue
            
            analyzer.add_user_message(message)
            analysis = analyzer.perform_analysis(message)
            
            # Display detailed results like sample conversation
            analyzer.display_detailed_analysis(analysis)
        
        analyzer.display_analysis_summary()
    
    # Save option
    save_choice = input("\n💾 Save session? (y/n): ").strip().lower()
    if save_choice == 'y':
        analyzer.save_session_data()
    
    print("\n👋 Thanks for using Ollama Trust & Sentiment Analyzer!")

if __name__ == "__main__":
    main()
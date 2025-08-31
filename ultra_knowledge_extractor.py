import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
import hashlib
from openai import OpenAI  # Add this import

class UltraKnowledgeExtractor:
    def __init__(self, kg_file="ultra_knowledge_graph.json", ollama_host="http://localhost:11434"):
        """Initialize the Ultra Knowledge Graph Extraction System with Ollama"""
        self.kg_file = kg_file
        self.ollama_host = ollama_host
        
        # Updated model definitions for Ollama
        self.available_models = {
            "1": {
                "name": "Llama-3.2-1B (Fastest)",
                "model_id": "llama3.2:1b",
                "type": "ollama"
            },
            "2": {
                "name": "Llama-3.2-3B (Balanced)", 
                "model_id": "llama3.2:3b",
                "type": "ollama"
            },
            "3": {
                "name": "Llama-3.1-8B (Best Quality)",
                "model_id": "llama3.1:8b",
                "type": "ollama"
            }
        }
        
        self.current_model = None
        self.ollama_client = None
        self.load_knowledge_graph()
        
    def setup_llm(self, model_choice: str):
        """Setup the selected LLM with Ollama"""
        if model_choice not in self.available_models:
            print("Invalid model choice!")
            return False
            
        model_info = self.available_models[model_choice]
        print(f"Setting up {model_info['name']} with Ollama...")
        
        try:
            # Initialize OpenAI client pointing to Ollama
            self.ollama_client = OpenAI(
                base_url=f"{self.ollama_host}/v1",
                api_key="ollama"  # Required but unused for Ollama
            )
            
            # Test connection by making a simple request
            test_response = self.ollama_client.chat.completions.create(
                model=model_info["model_id"],
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0.1
            )
            
            if test_response.choices:
                self.current_model = model_info
                print(f"✅ Successfully connected to {model_info['name']}")
                return True
            else:
                print(f"❌ No response from model {model_info['name']}")
                return False
                
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print(f"💡 Make sure Ollama is running: 'ollama serve'")
            print(f"💡 Make sure model is installed: 'ollama pull {model_info['model_id']}'")
            return False
    
    def generate_with_model(self, messages: List[Dict], max_tokens: int = 800):
        """Generate response using Ollama's OpenAI-compatible API - optimized for speed"""
        if not self.ollama_client or not self.current_model:
            return "No model loaded"
            
        try:
            response = self.ollama_client.chat.completions.create(
                model=self.current_model["model_id"],
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,  # Reduced for speed and consistency
                top_p=0.9,        # Added for speed
                stream=False      # Non-streaming for now
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error: {e}"
    
    def display_model_options(self):
        """Display available model options"""
        print("\n🤖 Available Ollama Models:")
        print("-" * 60)
        for key, model in self.available_models.items():
            print(f"  {key}. {model['name']}")
        print("-" * 60)
        print("\n💡 Make sure you've pulled the model first:")
        print("   ollama pull llama3.2:1b")
        print("   ollama pull llama3.2:3b") 
        print("   ollama pull llama3.1:8b")
    
    # All other methods remain exactly the same - no changes needed!
    def load_knowledge_graph(self):
        """Load or initialize the knowledge graph"""
        if os.path.exists(self.kg_file):
            with open(self.kg_file, 'r') as f:
                self.kg = json.load(f)
            print(f"📚 Loaded existing knowledge graph with {len(self.kg.get('entities', {}))} entities")
        else:
            self.kg = self.initialize_empty_kg()
            print("🆕 Initialized new knowledge graph")
            
    def initialize_empty_kg(self):
        """Initialize comprehensive knowledge graph structure"""
        return {
            # Core entities and their attributes
            "entities": {
                "user": {
                    "attributes": {},
                    "relationships": {},
                    "temporal_data": {},
                    "meta": {"created": str(datetime.now()), "last_updated": str(datetime.now())}
                }
            },
            
            # All extracted triplets with full context
            "triplets": [],
            
            # Relationship mappings for quick lookup
            "relationships": defaultdict(list),
            
            # Temporal patterns and changes
            "temporal_patterns": {},
            
            # Entity recognition (people, places, organizations, etc.)
            "named_entities": {
                "people": {},
                "places": {},
                "organizations": {},
                "events": {},
                "objects": {},
                "concepts": {}
            },
            
            # Conversation context for better extraction
            "conversation_metadata": {
                "sessions": [],
                "topics_discussed": [],
                "entity_mentions": defaultdict(int),
                "relationship_frequency": defaultdict(int)
            },
            
            # Enhanced categorization
            "knowledge_categories": {
                "identity": {},           # name, age, gender, nationality, etc.
                "demographics": {},       # location, education, occupation, etc.
                "physical": {},          # appearance, health, medical, etc.
                "psychological": {},     # personality, mental health, emotions, etc.
                "social": {},           # relationships, family, friends, etc.
                "professional": {},     # work, career, skills, etc.
                "lifestyle": {},        # habits, routines, preferences, etc.
                "interests": {},        # hobbies, entertainment, sports, etc.
                "goals": {},           # aspirations, plans, dreams, etc.
                "challenges": {},      # problems, stressors, fears, etc.
                "experiences": {},     # life events, memories, achievements, etc.
                "values": {},          # beliefs, principles, morals, etc.
                "financial": {},       # income, expenses, financial goals, etc.
                "technology": {},      # devices, apps, digital habits, etc.
                "communication": {},   # language preferences, communication style, etc.
                "temporal": {}         # schedules, time preferences, seasonal patterns, etc.
            },
            
            # Statistics and analytics
            "analytics": {
                "extraction_stats": {
                    "total_extractions": 0,
                    "successful_extractions": 0,
                    "failed_extractions": 0,
                    "avg_triplets_per_input": 0
                },
                "entity_growth": [],
                "knowledge_density": {}
            },
            
            "meta": {
                "version": "2.0",
                "created": str(datetime.now()),
                "last_updated": str(datetime.now()),
                "total_inputs_processed": 0
            }
        }
    
    def save_knowledge_graph(self):
        """Save knowledge graph with enhanced metadata"""
        self.kg["meta"]["last_updated"] = str(datetime.now())
        self.kg["meta"]["total_inputs_processed"] += 1
        
        # Update analytics
        total_triplets = len(self.kg["triplets"])
        total_inputs = self.kg["meta"]["total_inputs_processed"]
        if total_inputs > 0:
            self.kg["analytics"]["extraction_stats"]["avg_triplets_per_input"] = total_triplets / total_inputs
        
        with open(self.kg_file, 'w') as f:
            json.dump(self.kg, f, indent=2)
    
    def get_extraction_context(self, user_input: str) -> str:
        """Build comprehensive context for better extraction"""
        context_parts = []
        
        # Recent conversation topics
        if self.kg["conversation_metadata"]["topics_discussed"]:
            recent_topics = list(self.kg["conversation_metadata"]["topics_discussed"][-10:])
            context_parts.append(f"Recent topics discussed: {', '.join(recent_topics)}")
        
        # Frequently mentioned entities
        frequent_entities = sorted(
            self.kg["conversation_metadata"]["entity_mentions"].items(),
            key=lambda x: x[1], reverse=True
        )[:15]
        if frequent_entities:
            entities_str = ", ".join([f"{entity}({count})" for entity, count in frequent_entities])
            context_parts.append(f"Frequently mentioned entities: {entities_str}")
        
        # Known relationships
        known_relationships = []
        for rel_type, relationships in list(self.kg["relationships"].items())[:10]:
            for rel in relationships[:3]:
                known_relationships.append(f"{rel['subject']} {rel_type} {rel['object']}")
        if known_relationships:
            context_parts.append(f"Known relationships: {'; '.join(known_relationships)}")
        
        # Current knowledge in relevant categories
        input_keywords = set(user_input.lower().split())
        relevant_knowledge = []
        
        for category, knowledge in self.kg["knowledge_categories"].items():
            if not knowledge:
                continue
            for key, data in list(knowledge.items())[:5]:
                if isinstance(data, dict) and "current_value" in data:
                    value_words = set(str(data["current_value"]).lower().split())
                    if input_keywords.intersection(value_words) or any(keyword in key.lower() for keyword in input_keywords):
                        relevant_knowledge.append(f"{category}.{key}: {data['current_value']}")
        
        if relevant_knowledge:
            context_parts.append(f"Relevant existing knowledge: {'; '.join(relevant_knowledge[:10])}")
        
        return "\n".join(context_parts) if context_parts else "No existing context available."
    
    def ultra_extract_knowledge(self, user_input: str) -> Dict:
        """Ultra-comprehensive knowledge extraction - optimized for speed"""
        
        extraction_context = self.get_extraction_context(user_input)
        current_time = str(datetime.now())
        
        # Simplified and faster system prompt
        system_prompt = f"""You are a personal knowledge assistant. The user is voluntarily sharing information about themselves to build their personal knowledge graph. This is consensual information sharing for legitimate personal use.

CONTEXT: {extraction_context}

TASK: Extract factual information from the user's message and format as JSON. Focus on:
• Personal details (name, age, location, occupation)
• Relationships (family, friends, colleagues)
• Interests and activities
• Goals and preferences
• Any specific facts mentioned

IMPORTANT: This is the user's own information being stored in their personal system. Extract all relevant details.

Return JSON format:
{{
  "triplets": [
    {{
      "subject": "User", 
      "predicate": "relationship_or_attribute",
      "object": "value",
      "category": "identity|social|interests|goals|lifestyle",
      "confidence": 0.8-1.0
    }}
  ],
  "named_entities": {{
    "people": ["names"],
    "places": ["locations"], 
    "organizations": ["companies/schools"]
  }},
  "topics_mentioned": ["main topics"]
}}

Respond with JSON only, no other text."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'The user says: "{user_input}"\n\nExtract personal information and return as JSON only.'}
        ]
        
        try:
            response = self.generate_with_model(messages, max_tokens=800)  # Reduced for speed
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                extraction_json = json_match.group()
                extraction_data = json.loads(extraction_json)
                
                # Add metadata
                extraction_data["meta"] = {
                    "timestamp": current_time,
                    "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H')}",
                    "input_text": user_input,
                    "extraction_method": "ultra_comprehensive"
                }
                
                self.kg["analytics"]["extraction_stats"]["successful_extractions"] += 1
                return extraction_data
                
            else:
                print(f"⚠️ Could not extract JSON from response")
                self.kg["analytics"]["extraction_stats"]["failed_extractions"] += 1
                return {}
                
        except Exception as e:
            print(f"❌ Error in knowledge extraction: {e}")
            self.kg["analytics"]["extraction_stats"]["failed_extractions"] += 1
            return {}
    
    def integrate_extracted_knowledge(self, extracted_data: Dict, user_input: str):
        """Integrate extracted knowledge into the comprehensive knowledge graph"""
        if not extracted_data:
            return
        
        current_time = str(datetime.now())
        
        # Process triplets
        if "triplets" in extracted_data:
            for triplet in extracted_data["triplets"]:
                # Add to main triplets store
                triplet_with_meta = {
                    **triplet,
                    "id": self.generate_triplet_id(triplet),
                    "extracted_at": current_time,
                    "source_input": user_input
                }
                self.kg["triplets"].append(triplet_with_meta)
                
                # Add to relationships mapping
                predicate = triplet["predicate"]
                self.kg["relationships"][predicate].append({
                    "subject": triplet["subject"],
                    "object": triplet["object"],
                    "confidence": triplet.get("confidence", 0.8),
                    "timestamp": current_time,
                    "context": triplet.get("context", "")
                })
                
                # Add to knowledge categories
                category = triplet.get("category", "general")
                subcategory = triplet.get("subcategory", "general")
                
                if category not in self.kg["knowledge_categories"]:
                    self.kg["knowledge_categories"][category] = {}
                
                key = f"{triplet['predicate']}_{triplet['object']}".replace(" ", "_")
                
                self.kg["knowledge_categories"][category][key] = {
                    "subject": triplet["subject"],
                    "predicate": triplet["predicate"],
                    "object": triplet["object"],
                    "subcategory": subcategory,
                    "confidence": triplet.get("confidence", 0.8),
                    "temporal_info": triplet.get("temporal_info", "unknown"),
                    "last_updated": current_time,
                    "update_count": self.kg["knowledge_categories"][category].get(key, {}).get("update_count", 0) + 1,
                    "source_contexts": self.kg["knowledge_categories"][category].get(key, {}).get("source_contexts", []) + [triplet.get("context", "")]
                }
        
        # Process named entities
        if "named_entities" in extracted_data:
            for entity_type, entities in extracted_data["named_entities"].items():
                if entity_type not in self.kg["named_entities"]:
                    self.kg["named_entities"][entity_type] = {}
                
                for entity in entities:
                    if entity not in self.kg["named_entities"][entity_type]:
                        self.kg["named_entities"][entity_type][entity] = {
                            "first_mentioned": current_time,
                            "mention_count": 0,
                            "contexts": []
                        }
                    
                    self.kg["named_entities"][entity_type][entity]["mention_count"] += 1
                    self.kg["named_entities"][entity_type][entity]["contexts"].append({
                        "timestamp": current_time,
                        "input": user_input,
                        "context": extracted_data.get("context", "")
                    })
                    
                    # Update conversation metadata
                    self.kg["conversation_metadata"]["entity_mentions"][entity] += 1
        
        # Process other extracted information
        for key in ["topics_mentioned", "emotional_indicators", "temporal_markers", 
                   "relationship_dynamics", "behavior_patterns", "decision_factors", 
                   "communication_style", "priority_indicators"]:
            if key in extracted_data:
                if key not in self.kg["conversation_metadata"]:
                    self.kg["conversation_metadata"][key] = []
                
                for item in extracted_data[key]:
                    self.kg["conversation_metadata"][key].append({
                        "value": item,
                        "timestamp": current_time,
                        "source_input": user_input
                    })
        
        # Update analytics
        self.kg["analytics"]["extraction_stats"]["total_extractions"] += 1
        
        # Track knowledge growth
        entity_count = sum(len(entities) for entities in self.kg["named_entities"].values())
        self.kg["analytics"]["entity_growth"].append({
            "timestamp": current_time,
            "total_entities": entity_count,
            "total_triplets": len(self.kg["triplets"]),
            "total_categories": len([cat for cat in self.kg["knowledge_categories"] if self.kg["knowledge_categories"][cat]])
        })
    
    def generate_triplet_id(self, triplet: Dict) -> str:
        """Generate unique ID for triplet to avoid duplicates"""
        content = f"{triplet['subject']}_{triplet['predicate']}_{triplet['object']}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def extract_and_store(self, user_input: str) -> Dict:
        """Main method to extract and store knowledge from user input"""
        print("🧠 Ultra-extracting knowledge...")
        
        # Extract comprehensive knowledge
        extracted_data = self.ultra_extract_knowledge(user_input)
        
        if extracted_data:
            # Integrate into knowledge graph
            self.integrate_extracted_knowledge(extracted_data, user_input)
            
            # Save knowledge graph
            self.save_knowledge_graph()
            
            # Prepare summary for output
            summary = self.generate_extraction_summary(extracted_data)
            print(f"✅ Extracted and stored knowledge successfully")
            
            return {
                "extraction_successful": True,
                "extracted_triplets": len(extracted_data.get("triplets", [])),
                "named_entities_found": sum(len(entities) for entities in extracted_data.get("named_entities", {}).values()),
                "topics_identified": len(extracted_data.get("topics_mentioned", [])),
                "summary": summary,
                "total_knowledge_base_size": {
                    "total_triplets": len(self.kg["triplets"]),
                    "total_entities": sum(len(entities) for entities in self.kg["named_entities"].values()),
                    "knowledge_categories": len([cat for cat in self.kg["knowledge_categories"] if self.kg["knowledge_categories"][cat]])
                }
            }
        else:
            print("⚠️ No knowledge extracted from input")
            return {
                "extraction_successful": False,
                "message": "No extractable knowledge found in input"
            }
    
    def generate_extraction_summary(self, extracted_data: Dict) -> List[str]:
        """Generate human-readable summary of extracted information"""
        summary = []
        
        if "triplets" in extracted_data:
            triplets = extracted_data["triplets"]
            categories = defaultdict(int)
            for triplet in triplets:
                categories[triplet.get("category", "general")] += 1
            
            summary.append(f"Extracted {len(triplets)} knowledge triplets:")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                summary.append(f"  • {category}: {count} items")
        
        if "named_entities" in extracted_data:
            entities = extracted_data["named_entities"]
            total_entities = sum(len(ents) for ents in entities.values())
            if total_entities > 0:
                summary.append(f"Identified {total_entities} named entities:")
                for entity_type, entity_list in entities.items():
                    if entity_list:
                        summary.append(f"  • {entity_type}: {', '.join(entity_list[:3])}{'...' if len(entity_list) > 3 else ''}")
        
        return summary
    
    def display_knowledge_stats(self):
        """Display comprehensive knowledge graph statistics"""
        print("\n" + "="*100)
        print("📊 ULTRA KNOWLEDGE GRAPH STATISTICS")
        print("="*100)
        
        # Basic stats
        total_triplets = len(self.kg["triplets"])
        total_entities = sum(len(entities) for entities in self.kg["named_entities"].values())
        total_categories = len([cat for cat in self.kg["knowledge_categories"] if self.kg["knowledge_categories"][cat]])
        
        print(f"📈 Total Knowledge Triplets: {total_triplets}")
        print(f"👥 Total Named Entities: {total_entities}")
        print(f"📂 Active Knowledge Categories: {total_categories}")
        print(f"🔄 Total Inputs Processed: {self.kg['meta']['total_inputs_processed']}")
        print(f"🎯 Extraction Success Rate: {self.kg['analytics']['extraction_stats']['successful_extractions']}/{self.kg['analytics']['extraction_stats']['total_extractions']}")
        
        # Category breakdown
        print("\n📋 KNOWLEDGE BY CATEGORY:")
        print("-" * 80)
        for category, data in self.kg["knowledge_categories"].items():
            if data:
                count = len(data)
                print(f"  {category.title().replace('_', ' ')}: {count} items")
                
                # Show sample items
                sample_items = list(data.items())[:2]
                for key, info in sample_items:
                    obj = info.get("object", "unknown")
                    confidence = info.get("confidence", 0)
                    print(f"    • {info.get('predicate', 'has')}: {obj} (confidence: {confidence:.1f})")
        
        # Entity breakdown
        print("\n👥 NAMED ENTITIES:")
        print("-" * 80)
        for entity_type, entities in self.kg["named_entities"].items():
            if entities:
                count = len(entities)
                print(f"  {entity_type.title()}: {count}")
                # Show most mentioned
                sorted_entities = sorted(entities.items(), 
                                       key=lambda x: x[1]["mention_count"], reverse=True)
                for entity, data in sorted_entities[:3]:
                    mentions = data["mention_count"]
                    print(f"    • {entity} (mentioned {mentions} times)")
        
        # Recent growth
        if self.kg["analytics"]["entity_growth"]:
            latest_growth = self.kg["analytics"]["entity_growth"][-1]
            print(f"\n📈 Current Knowledge Graph Size:")
            print(f"  • Entities: {latest_growth['total_entities']}")
            print(f"  • Triplets: {latest_growth['total_triplets']}")
            print(f"  • Categories: {latest_growth['total_categories']}")
        
        print("="*100)
    
    def export_for_llm_context(self, max_triplets: int = 50, max_entities: int = 30) -> str:
        """Export knowledge graph in format optimized for LLM context"""
        context_parts = []
        
        # Recent high-confidence triplets
        recent_triplets = sorted(
            self.kg["triplets"],
            key=lambda x: (x.get("confidence", 0), x.get("extracted_at", "")),
            reverse=True
        )[:max_triplets]
        
        if recent_triplets:
            context_parts.append("KEY KNOWLEDGE ABOUT USER:")
            for triplet in recent_triplets:
                confidence = triplet.get("confidence", 0)
                if confidence >= 0.7:  # Only high-confidence info
                    subj = triplet["subject"]
                    pred = triplet["predicate"].replace("_", " ")
                    obj = triplet["object"]
                    temporal = triplet.get("temporal_info", "")
                    context_parts.append(f"• {subj} {pred} {obj} {f'({temporal})' if temporal else ''}")
        
        # Important entities
        important_entities = []
        for entity_type, entities in self.kg["named_entities"].items():
            sorted_entities = sorted(entities.items(), 
                                   key=lambda x: x[1]["mention_count"], reverse=True)
            for entity, data in sorted_entities[:5]:  # Top 5 per type
                if data["mention_count"] > 1:  # Only frequently mentioned
                    important_entities.append(f"{entity} ({entity_type})")
        
        if important_entities:
            context_parts.append(f"\nIMPORTANT ENTITIES: {', '.join(important_entities[:max_entities])}")
        
        # Behavioral patterns
        if "behavior_patterns" in self.kg["conversation_metadata"]:
            recent_patterns = [item["value"] for item in self.kg["conversation_metadata"]["behavior_patterns"][-10:]]
            if recent_patterns:
                context_parts.append(f"\nBEHAVIOR PATTERNS: {'; '.join(set(recent_patterns))}")
        
        return "\n".join(context_parts)

def main():
    """Main extraction loop"""
    print("🧠 ULTRA KNOWLEDGE EXTRACTION SYSTEM (Ollama Edition)")
    print("=" * 60)
    print("🎯 PURPOSE: Maximum information capture and knowledge graph building")
    print("💡 Optimized for feeding rich context to other LLMs")
    print("🔗 Powered by Ollama's OpenAI-compatible API")
    
    # Initialize the system
    extractor = UltraKnowledgeExtractor()
    
    # Model selection
    while True:
        extractor.display_model_options()
        print("\n📋 Recommendations:")
        print("  • Option 1: Fastest extraction (good for testing)")
        print("  • Option 2: Best balance (recommended for production)")
        print("  • Option 3: Highest quality extraction (most comprehensive)")
        
        model_choice = input("\nSelect a model (1-3): ").strip()
        
        if extractor.setup_llm(model_choice):
            break
        else:
            print("Please try again.")
    
    print("\n🎯 Ultra Knowledge Extractor is ready!")
    print("💡 Commands: 'quit' to exit, 'stats' for statistics, 'export' for LLM context")
    print("⚡ Commands: 'fast' to toggle fast mode, 'full' to toggle full mode")
    print("🔍 Share anything and I'll extract maximum knowledge for your knowledge graph")
    print("-" * 80)
    
    # Speed mode toggle
    fast_mode = True
    print(f"🚀 Current mode: {'FAST' if fast_mode else 'COMPREHENSIVE'}")
    
    # Main extraction loop
    while True:
        user_input = input("\nInput: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("\n👋 Knowledge extraction session complete!")
            break
        elif user_input.lower() == 'stats':
            extractor.display_knowledge_stats()
            continue
        elif user_input.lower() == 'export':
            context = extractor.export_for_llm_context()
            print("\n📤 EXPORTED CONTEXT FOR LLM:")
            print("-" * 60)
            print(context)
            print("-" * 60)
            continue
        elif user_input.lower() == 'fast':
            fast_mode = True
            print("🚀 Switched to FAST mode (optimized for speed)")
            continue
        elif user_input.lower() == 'full':
            fast_mode = False
            print("🔍 Switched to COMPREHENSIVE mode (detailed extraction)")
            continue
        elif not user_input:
            continue
        
        # Extract and store knowledge
        try:
            import time
            start_time = time.time()
            
            # Toggle extraction mode based on fast_mode
            if fast_mode and hasattr(extractor, 'response_cache'):
                # Force quick mode
                old_method = extractor.extract_and_store
                def quick_wrapper(user_input):
                    if len(user_input.split()) < 20:
                        return old_method(user_input)
                    else:
                        # For longer inputs in fast mode, still use quick extraction
                        extracted_data = extractor.quick_extract_knowledge(user_input)
                        if extracted_data:
                            extractor.integrate_extracted_knowledge(extracted_data, user_input)
                            extractor.save_knowledge_graph()
                            summary = extractor.generate_extraction_summary(extracted_data)
                            return {
                                "extraction_successful": True,
                                "extracted_triplets": len(extracted_data.get("triplets", [])),
                                "named_entities_found": sum(len(entities) for entities in extracted_data.get("named_entities", {}).values()),
                                "topics_identified": len(extracted_data.get("topics_mentioned", [])),
                                "summary": summary,
                                "total_knowledge_base_size": {
                                    "total_triplets": len(extractor.kg["triplets"]),
                                    "total_entities": sum(len(entities) for entities in extractor.kg["named_entities"].values()),
                                    "knowledge_categories": len([cat for cat in extractor.kg["knowledge_categories"] if extractor.kg["knowledge_categories"][cat]])
                                }
                            }
                        else:
                            return {"extraction_successful": False, "message": "No extractable knowledge found"}
                
                result = quick_wrapper(user_input)
            else:
                # Use normal extraction
                result = extractor.extract_and_store(user_input)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if result["extraction_successful"]:
                print(f"\n✅ EXTRACTION COMPLETE ({processing_time:.2f}s):")
                for line in result["summary"]:
                    print(f"   {line}")
                
                total = result["total_knowledge_base_size"]
                print(f"\n📊 Knowledge Base: {total['total_triplets']} triplets, {total['total_entities']} entities, {total['knowledge_categories']} categories")
            else:
                print(f"\n⚠️ {result['message']} ({processing_time:.2f}s)")
                
        except Exception as e:
            print(f"\n❌ Error during extraction: {e}")

if __name__ == "__main__":
    main()
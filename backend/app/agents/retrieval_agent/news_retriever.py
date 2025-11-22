import requests
import feedparser
import json
from datetime import datetime
from typing import List, Dict
import time
import random

class NewsRetriever:
    def __init__(self):
        self.sources = {
            'WHO': 'https://www.who.int/feeds/entity/csr/don/en/rss.xml',
            'CDC': 'https://tools.cdc.gov/api/v2/resources/news',
            'NIH': 'https://www.nih.gov/news-events/news-releases/rss',
        }
        self.cache = {}
        self.cache_timeout = 3600  # 1 hour cache
        
        # Health-related images from Unsplash (free to use)
        self.health_images = {
            'covid': 'https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?w=400&h=250&fit=crop',  # COVID prevention
            'vaccination': 'https://images.unsplash.com/photo-1631941618536-2979d565b726?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8dmFjY2luYXRpb258ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600',  # Vaccination
            'mental_health': 'https://images.unsplash.com/photo-1590650153855-d9e808231d41?w=400&h=250&fit=crop',  # Mental health
            'nutrition': 'https://images.unsplash.com/photo-1490645935967-10de6ba17061?w=400&h=250&fit=crop',  # Nutrition
            'exercise': 'https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?w=400&h=250&fit=crop',  # Exercise
            'flu': 'https://media.istockphoto.com/id/2159146285/photo/young-woman-feeling-unwell-at-home-sitting-on-couch-with-tissue-and-blanket-battling-cold-or.webp?a=1&b=1&s=612x612&w=0&k=20&c=9SAgETQU-EqUVt1bJFrXc6I79wFoF1KMhhDUuW3Hvhg=',  # Flu prevention
            'diabetes': 'https://images.unsplash.com/photo-1683727186226-910f31a9da45?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8ZGlhYmV0aWN8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600',  # Diabetes
            'dengue': 'https://media.istockphoto.com/id/486376513/photo/mosquito-sucking-blood_set-b-4.webp?a=1&b=1&s=612x612&w=0&k=20&c=Exv5fl8vtHNIov4vVbTZMtWlBQ3sOnmqNRkB9k5EhG4=',  # Mosquito/dengue
            'sleep': 'https://images.unsplash.com/photo-1541781774459-bb2af2f05b55?w=400&h=250&fit=crop',  # Sleep health
            'child_health': 'https://images.unsplash.com/photo-1537368910025-700350fe46c7?w=400&h=250&fit=crop',  # Child health
            'antibiotics': 'https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?w=400&h=250&fit=crop',  # Medication
            'stress': 'https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?w=400&h=250&fit=crop'  # Stress relief
        }
    
    def fetch_health_news(self, limit: int = 12) -> List[Dict]:
        """Fetch health news from various sources with fallback"""
        # Check cache first
        cache_key = f"news_{limit}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_timeout:
                return cached_data[:limit]
        
        all_news = []
        
        try:
            # Try to fetch from real sources first
            print("Fetching health news from real sources...")
            who_news = self._fetch_who_news()
            all_news.extend(who_news)
            
            cdc_news = self._fetch_cdc_news()
            all_news.extend(cdc_news)
            
            nih_news = self._fetch_nih_news()
            all_news.extend(nih_news)
            
            # If we got real news, use it
            if len(all_news) > 0:
                print(f"Successfully fetched {len(all_news)} real news items")
            else:
                # If no real news, use our educational health content
                print("Using educational health content...")
                all_news = self._get_educational_health_content()
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            # Use educational content as fallback
            all_news = self._get_educational_health_content()
        
        # Remove duplicates based on title
        unique_news = self._remove_duplicates(all_news)
        
        # Sort by date (newest first) if we have real news
        if len(who_news) > 0 or len(cdc_news) > 0 or len(nih_news) > 0:
            sorted_news = sorted(unique_news, key=lambda x: x.get('published', ''), reverse=True)
        else:
            sorted_news = unique_news
        
        # Cache the results
        self.cache[cache_key] = (time.time(), sorted_news)
        
        return sorted_news[:limit]
    
    def _fetch_who_news(self) -> List[Dict]:
        """Fetch news from WHO RSS feed"""
        news_items = []
        try:
            feed = feedparser.parse(self.sources['WHO'])
            for entry in feed.entries[:10]:
                news_items.append({
                    'title': entry.title,
                    'summary': entry.summary if hasattr(entry, 'summary') else entry.title,
                    'link': entry.link,
                    'published': entry.published if hasattr(entry, 'published') else datetime.now().isoformat(),
                    'source': 'World Health Organization',
                    'category': 'Public Health',
                    'type': 'outbreak_news',
                    'image_url': self._get_image_for_category('Public Health')
                })
        except Exception as e:
            print(f"Error fetching WHO news: {e}")
        return news_items
    
    def _fetch_cdc_news(self) -> List[Dict]:
        """Fetch news from CDC API"""
        news_items = []
        try:
            response = requests.get(self.sources['CDC'], params={'limit': 10}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('results', [])[:10]:
                    news_items.append({
                        'title': item.get('title', ''),
                        'summary': item.get('description', 'No description available'),
                        'link': item.get('link', ''),
                        'published': item.get('pubDate', datetime.now().isoformat()),
                        'source': 'Centers for Disease Control',
                        'category': 'Health Alert',
                        'type': 'health_alert',
                        'image_url': self._get_image_for_category('Health Alert')
                    })
        except Exception as e:
            print(f"Error fetching CDC news: {e}")
        return news_items
    
    def _fetch_nih_news(self) -> List[Dict]:
        """Fetch news from NIH RSS feed"""
        news_items = []
        try:
            feed = feedparser.parse(self.sources['NIH'])
            for entry in feed.entries[:10]:
                news_items.append({
                    'title': entry.title,
                    'summary': entry.summary if hasattr(entry, 'summary') else entry.title,
                    'link': entry.link,
                    'published': entry.published if hasattr(entry, 'published') else datetime.now().isoformat(),
                    'source': 'National Institutes of Health',
                    'category': 'Research News',
                    'type': 'research',
                    'image_url': self._get_image_for_category('Research')
                })
        except Exception as e:
            print(f"Error fetching NIH news: {e}")
        return news_items
    
    def _get_image_for_category(self, category: str) -> str:
        """Get appropriate image based on category"""
        category_map = {
            'Infectious Diseases': 'covid',
            'Preventive Care': 'vaccination',
            'Mental Wellness': 'mental_health',
            'Nutrition': 'nutrition',
            'Fitness': 'exercise',
            'Seasonal Health': 'flu',
            'Chronic Disease': 'diabetes',
            'Vector Diseases': 'dengue',
            'Wellness': 'sleep',
            'Child Health': 'child_health',
            'Public Health': 'antibiotics',
            'Stress Relief': 'stress'
        }
        
        image_key = category_map.get(category, 'covid')
        return self.health_images.get(image_key, self.health_images['covid'])
    
    def _get_educational_health_content(self) -> List[Dict]:
        """Get real educational health content that works offline - Exactly 12 items with images"""
        current_time = datetime.now().isoformat()
        
        # 12 carefully ordered health topics with working links and images
        health_content = [
            {
                'title': 'COVID-19 Prevention Guidelines',
                'summary': 'Regular hand washing, mask wearing in crowded places, and vaccination remain the most effective ways to prevent COVID-19 transmission according to health authorities.',
                'link': 'https://www.who.int/emergencies/diseases/novel-coronavirus-2019',
                'published': current_time,
                'source': 'WHO Health Guidelines',
                'category': 'Infectious Diseases',
                'type': 'prevention',
                'image_url': self.health_images['covid']
            },
            {
                'title': 'Vaccination Schedule Updates',
                'summary': 'Stay updated with recommended vaccination schedules for children and adults to maintain immunity against preventable diseases.',
                'link': 'https://www.cdc.gov/vaccines/?CDC_AAref_Val=https://www.cdc.gov/vaccines/schedules/',
                'published': current_time,
                'source': 'CDC Immunization',
                'category': 'Preventive Care',
                'type': 'updates',
                'image_url': self.health_images['vaccination']
            },
            {
                'title': 'Mental Health Support Resources',
                'summary': 'Access to mental health services is crucial. Many organizations offer free counseling and support for anxiety, depression, and stress management.',
                'link': 'https://www.who.int/health-topics/mental-health',
                'published': current_time,
                'source': 'WHO Mental Health',
                'category': 'Mental Wellness',
                'type': 'resources',
                'image_url': self.health_images['mental_health']
            },
            {
                'title': 'Healthy Nutrition Guidelines',
                'summary': 'Balanced diets rich in fruits, vegetables, and whole grains help maintain optimal health and prevent chronic diseases like diabetes and heart conditions.',
                'link': 'https://www.who.int/health-topics/healthy-diet',
                'published': current_time,
                'source': 'WHO Nutrition',
                'category': 'Nutrition',
                'type': 'guidelines',
                'image_url': self.health_images['nutrition']
            },
            {
                'title': 'Exercise and Physical Activity',
                'summary': 'Regular physical activity improves cardiovascular health, strengthens immune system, and reduces risk of many chronic diseases.',
                'link': 'https://www.who.int/news-room/fact-sheets/detail/physical-activity',
                'published': current_time,
                'source': 'WHO Physical Health',
                'category': 'Fitness',
                'type': 'recommendations',
                'image_url': self.health_images['exercise']
            },
            {
                'title': 'Seasonal Flu Prevention',
                'summary': 'Annual flu vaccination, proper hand hygiene, and avoiding close contact with sick individuals help prevent seasonal influenza transmission.',
                'link': 'https://www.cdc.gov/flu',
                'published': current_time,
                'source': 'CDC Influenza',
                'category': 'Seasonal Health',
                'type': 'prevention',
                'image_url': self.health_images['flu']
            },
            {
                'title': 'Diabetes Management Tips',
                'summary': 'Regular blood sugar monitoring, balanced diet, and physical activity are key to managing diabetes and preventing complications.',
                'link': 'https://www.who.int/health-topics/diabetes',
                'published': current_time,
                'source': 'WHO Diabetes',
                'category': 'Chronic Disease',
                'type': 'management',
                'image_url': self.health_images['diabetes']
            },
            {
                'title': 'Dengue Fever Awareness',
                'summary': 'Dengue cases typically increase during rainy seasons. Health officials recommend eliminating stagnant water sources and using mosquito repellents.',
                'link': 'https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue',
                'published': current_time,
                'source': 'WHO Disease Control',
                'category': 'Vector Diseases',
                'type': 'awareness',
                'image_url': self.health_images['dengue']
            },
            {
                'title': 'Sleep and Health Connection',
                'summary': 'Adequate sleep is essential for physical and mental health. Adults should aim for 7-9 hours of quality sleep per night for optimal functioning.',
                'link': 'https://www.cdc.gov/sleep',
                'published': current_time,
                'source': 'CDC Sleep Health',
                'category': 'Wellness',
                'type': 'health_tips',
                'image_url': self.health_images['sleep']
            },
            {
                'title': 'Childhood Immunization Importance',
                'summary': 'Vaccinating children according to recommended schedules protects them from serious diseases and helps maintain community immunity.',
                'link': 'https://www.who.int/teams/immunization-vaccines-and-biologicals',
                'published': current_time,
                'source': 'WHO Pediatrics',
                'category': 'Child Health',
                'type': 'immunization',
                'image_url': self.health_images['child_health']
            },
            {
                'title': 'Antibiotic Resistance Concerns',
                'summary': 'Misuse of antibiotics contributes to drug resistance. Only use antibiotics when prescribed and complete the full course as directed.',
                'link': 'https://www.who.int/health-topics/antimicrobial-resistance',
                'published': current_time,
                'source': 'WHO Medication Safety',
                'category': 'Public Health',
                'type': 'advisory',
                'image_url': self.health_images['antibiotics']
            },
            {
                'title': 'Stress Management Techniques',
                'summary': 'Regular exercise, mindfulness practices, and maintaining social connections help manage stress and promote mental wellbeing.',
                'link': 'https://www.who.int/news-room/questions-and-answers/item/stress',
                'published': current_time,
                'source': 'WHO Mental Wellness',
                'category': 'Stress Relief',
                'type': 'techniques',
                'image_url': self.health_images['stress']
            }
        ]
        
        return health_content
    
    def _remove_duplicates(self, news_list: List[Dict]) -> List[Dict]:
        """Remove duplicate news items based on title"""
        seen_titles = set()
        unique_news = []
        
        for item in news_list:
            title = item.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(item)
        
        return unique_news

# For testing
def main():
    retriever = NewsRetriever()
    news = retriever.fetch_health_news(limit=12)
    print(f"Fetched {len(news)} news items")
    for i, item in enumerate(news):
        print(f"{i+1}. {item['title']} - Image: {item.get('image_url', 'No image')}")

if __name__ == "__main__":
    main()
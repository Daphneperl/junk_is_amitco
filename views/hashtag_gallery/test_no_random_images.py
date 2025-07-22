#!/usr/bin/env python3
"""
Test script to verify that no quotes are getting random images anymore
"""

import json
from semantic_hashtag_connector import SemanticHashtagConnector

def test_no_random_images():
    """Test that no quotes are getting random images"""
    
    # Load the connections
    with open('quote_to_images_connections.json', 'r', encoding='utf-8') as f:
        connections = json.load(f)
    
    print("Testing for Random Images in All Quotes")
    print("=" * 60)
    
    random_quotes = []
    low_score_quotes = []
    
    for quote, data in connections.items():
        has_random = False
        has_low_scores = False
        
        for detail in data['matching_details']:
            reason = detail['reason']
            score = detail['score']
            
            # Check for random fallback
            if 'random' in reason.lower() or 'Random selection' in reason:
                has_random = True
            
            # Check for very low scores (potential random matches)
            if score < 1.0:
                has_low_scores = True
        
        if has_random:
            random_quotes.append(quote)
        elif has_low_scores:
            low_score_quotes.append(quote)
    
    print(f"Quotes with random images: {len(random_quotes)}")
    if random_quotes:
        print("Random quotes found:")
        for quote in random_quotes[:10]:  # Show first 10
            print(f"  - {quote}")
        if len(random_quotes) > 10:
            print(f"  ... and {len(random_quotes) - 10} more")
    else:
        print("✅ No quotes with random images found!")
    
    print(f"\nQuotes with low scores (< 1.0): {len(low_score_quotes)}")
    if low_score_quotes:
        print("Low score quotes:")
        for quote in low_score_quotes[:10]:  # Show first 10
            print(f"  - {quote}")
        if len(low_score_quotes) > 10:
            print(f"  ... and {len(low_score_quotes) - 10} more")
    else:
        print("✅ No quotes with low scores found!")
    
    return len(random_quotes) == 0

def show_improved_matches():
    """Show examples of improved matches"""
    
    # Load the connections
    with open('quote_to_images_connections.json', 'r', encoding='utf-8') as f:
        connections = json.load(f)
    
    print(f"\nExamples of Improved Matches")
    print("=" * 60)
    
    # Show some examples of quotes that previously had random images
    test_quotes = [
        "Kill_me_softly_with_science",
        "This_page_intentionally_left_blank", 
        "You_wouldnt_download_afriend",
        "Abort_retry_fail",
        "We_are_the_dead_pixels"
    ]
    
    for quote in test_quotes:
        if quote in connections:
            data = connections[quote]
            print(f"\nQuote: {quote}")
            print(f"Bottom Line: {data['bottom_line']}")
            print(f"Top 5 matches:")
            
            for i, detail in enumerate(data['matching_details'][:5], 1):
                print(f"  {i}. {detail['filename']} (Score: {detail['score']:.1f}) - {detail['reason']}")

def analyze_score_distribution():
    """Analyze the distribution of scores across all quotes"""
    
    # Load the connections
    with open('quote_to_images_connections.json', 'r', encoding='utf-8') as f:
        connections = json.load(f)
    
    print(f"\nScore Distribution Analysis")
    print("=" * 60)
    
    all_scores = []
    score_ranges = {
        '0-1': 0,
        '1-5': 0,
        '5-10': 0,
        '10-20': 0,
        '20+': 0
    }
    
    for quote, data in connections.items():
        for detail in data['matching_details']:
            score = detail['score']
            all_scores.append(score)
            
            if score < 1:
                score_ranges['0-1'] += 1
            elif score < 5:
                score_ranges['1-5'] += 1
            elif score < 10:
                score_ranges['5-10'] += 1
            elif score < 20:
                score_ranges['10-20'] += 1
            else:
                score_ranges['20+'] += 1
    
    total_matches = len(all_scores)
    print(f"Total image matches: {total_matches}")
    print(f"Average score: {sum(all_scores) / total_matches:.2f}")
    print(f"Score distribution:")
    for range_name, count in score_ranges.items():
        percentage = (count / total_matches) * 100
        print(f"  {range_name}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    no_random = test_no_random_images()
    show_improved_matches()
    analyze_score_distribution()
    
    if no_random:
        print(f"\n🎉 SUCCESS: All quotes now have semantic matches!")
    else:
        print(f"\n⚠️  WARNING: Some quotes still have random images") 
#!/usr/bin/env python3
"""
Test script to demonstrate the bottom line matching improvements
"""

import json
from semantic_hashtag_connector import SemanticHashtagConnector

def test_bottom_line_matching():
    """Test and demonstrate bottom line matching improvements"""
    
    # Initialize the connector
    connector = SemanticHashtagConnector()
    connector.load_data()
    
    # Test cases with different bottom lines
    test_cases = [
        {
            'quote': 'Sometimes_dead_is_better',
            'bottom_line': 'Warning',
            'expected_keywords': ['warning', 'danger', 'caution', 'alert', 'threat', 'risk']
        },
        {
            'quote': 'Hell_is_ateenage_girl',
            'bottom_line': 'Venom',
            'expected_keywords': ['venom', 'poison', 'toxic', 'deadly', 'lethal', 'harmful']
        },
        {
            'quote': 'We_were_never_in_control',
            'bottom_line': 'Chaos',
            'expected_keywords': ['chaos', 'disorder', 'confusion', 'random', 'unpredictable']
        },
        {
            'quote': 'Time_is_a_flat_circle',
            'bottom_line': 'Recurrence',
            'expected_keywords': ['recurrence', 'repetition', 'cycle', 'loop', 'return', 'repeat']
        },
        {
            'quote': 'Love_is_the_drug_and_iwont_give_up',
            'bottom_line': 'Addiction',
            'expected_keywords': ['addiction', 'obsession', 'compulsion', 'dependence', 'habit']
        }
    ]
    
    print("Testing Bottom Line Matching Improvements")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Quote: {test_case['quote']}")
        print(f"   Bottom Line: {test_case['bottom_line']}")
        print(f"   Expected Keywords: {', '.join(test_case['expected_keywords'])}")
        
        # Get the bottom line keywords from our system
        if test_case['bottom_line'] in connector.bottom_line_keywords:
            system_keywords = connector.bottom_line_keywords[test_case['bottom_line']]
            print(f"   System Keywords: {', '.join(system_keywords)}")
            
            # Find matches with bottom line emphasis
            matches = connector.find_best_matches_with_explanations(
                test_case['quote'], 
                num_matches=5, 
                bottom_line=test_case['bottom_line']
            )
            
            print(f"   Top 5 Matches:")
            for j, (filename, score, explanation) in enumerate(matches, 1):
                reason = "No specific reason found"
                if explanation.get('bottom_line_matches'):
                    reason = explanation['bottom_line_matches'][0]
                elif explanation.get('direct_matches'):
                    reason = explanation['direct_matches'][0]
                elif explanation.get('special_connections'):
                    reason = explanation['special_connections'][0]
                
                print(f"     {j}. {filename} (Score: {score:.1f}) - {reason}")
        else:
            print(f"   ERROR: Bottom line '{test_case['bottom_line']}' not found in system")
        
        print("-" * 60)

def compare_with_and_without_bottom_line():
    """Compare matching results with and without bottom line emphasis"""
    
    connector = SemanticHashtagConnector()
    connector.load_data()
    
    test_quote = "We_were_never_in_control"
    test_bottom_line = "Chaos"
    
    print(f"\nComparison: With vs Without Bottom Line Emphasis")
    print(f"Quote: {test_quote}")
    print(f"Bottom Line: {test_bottom_line}")
    print("=" * 80)
    
    # Get matches WITHOUT bottom line emphasis
    matches_without = connector.find_best_matches_with_explanations(
        test_quote, 
        num_matches=5
    )
    
    # Get matches WITH bottom line emphasis
    matches_with = connector.find_best_matches_with_explanations(
        test_quote, 
        num_matches=5, 
        bottom_line=test_bottom_line
    )
    
    print(f"\nWithout Bottom Line Emphasis:")
    for i, (filename, score, explanation) in enumerate(matches_without, 1):
        reason = "No specific reason found"
        if explanation.get('direct_matches'):
            reason = explanation['direct_matches'][0]
        elif explanation.get('special_connections'):
            reason = explanation['special_connections'][0]
        print(f"  {i}. {filename} (Score: {score:.1f}) - {reason}")
    
    print(f"\nWith Bottom Line Emphasis:")
    for i, (filename, score, explanation) in enumerate(matches_with, 1):
        reason = "No specific reason found"
        if explanation.get('bottom_line_matches'):
            reason = explanation['bottom_line_matches'][0]
        elif explanation.get('direct_matches'):
            reason = explanation['direct_matches'][0]
        elif explanation.get('special_connections'):
            reason = explanation['special_connections'][0]
        print(f"  {i}. {filename} (Score: {score:.1f}) - {reason}")

if __name__ == "__main__":
    test_bottom_line_matching()
    compare_with_and_without_bottom_line() 
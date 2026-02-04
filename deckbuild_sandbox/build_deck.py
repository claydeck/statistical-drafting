#!/usr/bin/env python3
"""
Deck Builder CLI Tool

Build optimal decks from card pools using trained deckbuild models.
Supports MTG Arena format for easy copy-paste from the game.

Usage:
    # From a text file (supports Arena format)
    python build_deck.py --set TLA --pool-file my_pool.txt

    # From command line (comma-separated)
    python build_deck.py --set TLA --pool "Card A, Card A, Card B, Card C"

    # Interactive mode (paste from Arena)
    python build_deck.py --set TLA --interactive

MTG Arena format supported:
    2 Card Name (TLA) 123
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Optional

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statisticaldeckbuild as sdb


def parse_arena_card_line(line: str) -> tuple:
    """
    Parse a single line in MTG Arena format.

    Supports:
    - "Card Name"
    - "2 Card Name"
    - "2x Card Name"
    - "2 Card Name (SET) 123"  (Arena export format)

    Returns:
        Tuple of (quantity, card_name) or (None, None) if not a card line.
    """
    line = line.strip()

    # Skip empty lines, comments, and section headers
    if not line or line.startswith('#'):
        return None, None

    # Skip Arena section headers
    if line.lower() in ['deck', 'sideboard', 'commander', 'companion']:
        return None, None

    # Remove Arena set/collector number suffix: "(SET) 123" or "(SET)"
    # Pattern: text (XXX) optional_number
    import re
    arena_suffix = re.search(r'\s+\([A-Z0-9]{2,5}\)\s*\d*\s*$', line)
    if arena_suffix:
        line = line[:arena_suffix.start()]

    # Check for quantity prefix: "2 Card Name" or "2x Card Name"
    parts = line.split(maxsplit=1)
    if len(parts) == 2:
        first_part = parts[0].lower().rstrip('x')
        if first_part.isdigit():
            quantity = int(first_part)
            card_name = parts[1].strip()
            return quantity, card_name

    # No quantity prefix, assume 1 copy
    return 1, line


def load_pool_from_file(file_path: str) -> List[str]:
    """
    Load card pool from a text file.

    Supports formats:
    - One card per line: "Card Name"
    - With quantity: "2 Card Name" or "2x Card Name"
    - MTG Arena format: "2 Card Name (SET) 123"
    - Comments starting with # are ignored
    - Empty lines are ignored
    - Section headers (Deck, Sideboard) are ignored
    """
    pool = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            quantity, card_name = parse_arena_card_line(line)
            if card_name:
                pool.extend([card_name] * quantity)

    return pool


def parse_pool_from_string(pool_string: str) -> List[str]:
    """
    Parse card pool from a comma-separated string.

    Supports:
    - "Card A, Card B, Card C"
    - "2x Card A, Card B, 3x Card C"
    - "2 Card A (SET) 123, Card B"  (Arena format)
    """
    pool = []

    for item in pool_string.split(','):
        quantity, card_name = parse_arena_card_line(item)
        if card_name:
            pool.extend([card_name] * quantity)

    return pool


def interactive_pool_input() -> List[str]:
    """
    Interactively collect card pool from user input.
    Supports MTG Arena format (paste directly from Arena).
    """
    print("\nEnter your card pool (one card per line).")
    print("Supports MTG Arena format: '2 Card Name (SET) 123'")
    print("Type 'done' or press Enter twice to finish.\n")

    pool = []
    empty_count = 0

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break

        if line.lower() == 'done':
            break

        if not line:
            empty_count += 1
            if empty_count >= 2:
                break
            continue

        empty_count = 0

        # Parse using Arena-compatible parser
        quantity, card_name = parse_arena_card_line(line)
        if card_name:
            pool.extend([card_name] * quantity)
            print(f"  Added: {quantity}x {card_name}")
        else:
            print(f"  Skipped: {line}")

    return pool


def export_deck_to_file(result: Dict, output_path: str, format: str = "txt") -> None:
    """
    Export deck to a file.

    Args:
        result: Result dictionary from build_deck()
        output_path: Output file path
        format: Output format ("txt", "json", "arena")
    """
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    elif format == "arena":
        # MTG Arena format
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Deck\n")
            for card_name, qty in sorted(result["deck_counts"].items()):
                f.write(f"{qty} {card_name}\n")
            f.write("\nSideboard\n")
            for card_name, qty in sorted(result["sideboard_counts"].items()):
                f.write(f"{qty} {card_name}\n")

    else:  # txt format
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Deck ({len(result['deck'])} cards)\n")
            for card_name, qty in sorted(result["deck_counts"].items(), key=lambda x: -result["scores"][x[0]]):
                score = result["scores"][card_name]
                f.write(f"{qty}x {card_name}  (score: {score:.1f})\n")

            f.write(f"\n# Sideboard ({len(result['sideboard'])} cards)\n")
            for card_name, qty in sorted(result["sideboard_counts"].items(), key=lambda x: -result["scores"][x[0]]):
                score = result["scores"][card_name]
                f.write(f"{qty}x {card_name}  (score: {score:.1f})\n")

    print(f"Exported to {output_path}")


def build_deck_cli(
    set_abbreviation: str,
    draft_mode: str,
    pool: List[str],
    target_deck_size: int = 23,
    verbose: bool = False,
    output_file: Optional[str] = None,
    output_format: str = "txt",
    model_folder: str = "../data/models/",
    cards_folder: str = "../data/cards/",
) -> Dict:
    """
    Build a deck from a card pool using the CLI.

    Args:
        set_abbreviation: Set code (e.g., "TLA")
        draft_mode: Draft mode ("Premier", "Trad", etc.)
        pool: List of card names
        target_deck_size: Number of non-land cards
        verbose: Show detailed progress
        output_file: Optional file to export results
        output_format: Export format ("txt", "json", "arena")
        model_folder: Path to model weights
        cards_folder: Path to card data

    Returns:
        Result dictionary from build_deck()
    """
    # Validate pool
    if not pool:
        print("Error: Empty card pool")
        sys.exit(1)

    print(f"\nDeck Builder")
    print("=" * 50)
    print(f"Set: {set_abbreviation}")
    print(f"Mode: {draft_mode}")
    print(f"Pool size: {len(pool)} cards")
    print(f"Target deck size: {target_deck_size} non-land cards")
    print("=" * 50)

    # Initialize builder
    try:
        builder = sdb.IterativeDeckBuilder(
            set_abbreviation=set_abbreviation,
            draft_mode=draft_mode,
            model_folder=model_folder,
            cards_folder=cards_folder,
        )
        print(f"Loaded model: {set_abbreviation}_{draft_mode}_deckbuild.pt")
        print(f"Device: {builder.device}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Build deck
    print("\nBuilding deck...")
    result = builder.build_deck(
        pool=pool,
        target_deck_size=target_deck_size,
        verbose=verbose,
    )

    # Print results
    builder.print_deck_and_sideboard(result, pool)

    # Export if requested
    if output_file:
        export_deck_to_file(result, output_file, output_format)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build optimal decks from card pools using trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From a text file (supports Arena format)
  python build_deck.py --set TLA --pool-file my_pool.txt

  # From command line
  python build_deck.py --set TLA --pool "2x Card A, Card B, 3x Card C"

  # Interactive mode (paste from Arena)
  python build_deck.py --set TLA --interactive

  # Export to file
  python build_deck.py --set TLA --pool-file pool.txt --output deck.txt

  # Export in MTG Arena format
  python build_deck.py --set TLA --pool-file pool.txt --output deck.txt --format arena

Pool file format (all formats supported):
  # Comments start with #
  Card Name
  2x Card Name
  2 Card Name
  2 Card Name (SET) 123    <-- MTG Arena format
        """
    )

    # Required arguments
    parser.add_argument(
        "--set", "-s",
        type=str,
        required=True,
        help="Set abbreviation (e.g., TLA, FDN, DSK)"
    )

    # Pool input (one of these is required)
    pool_group = parser.add_mutually_exclusive_group(required=True)
    pool_group.add_argument(
        "--pool-file", "-f",
        type=str,
        help="Path to text file containing card pool"
    )
    pool_group.add_argument(
        "--pool", "-p",
        type=str,
        help="Comma-separated list of cards (e.g., '2x Card A, Card B')"
    )
    pool_group.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enter card pool interactively"
    )

    # Optional arguments
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="Premier",
        choices=["Premier", "Trad", "PickTwo"],
        help="Draft mode (default: Premier)"
    )
    parser.add_argument(
        "--deck-size", "-d",
        type=int,
        default=23,
        help="Target deck size (non-land cards, default: 23)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed build progress"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="txt",
        choices=["txt", "json", "arena"],
        help="Output format (default: txt)"
    )
    parser.add_argument(
        "--model-folder",
        type=str,
        default="../data/models/",
        help="Path to model weights folder"
    )
    parser.add_argument(
        "--cards-folder",
        type=str,
        default="../data/cards/",
        help="Path to card data folder"
    )

    args = parser.parse_args()

    # Get pool from specified source
    if args.pool_file:
        if not os.path.exists(args.pool_file):
            print(f"Error: Pool file not found: {args.pool_file}")
            sys.exit(1)
        pool = load_pool_from_file(args.pool_file)
        print(f"Loaded {len(pool)} cards from {args.pool_file}")

    elif args.pool:
        pool = parse_pool_from_string(args.pool)

    elif args.interactive:
        pool = interactive_pool_input()

    # Build deck
    build_deck_cli(
        set_abbreviation=args.set,
        draft_mode=args.mode,
        pool=pool,
        target_deck_size=args.deck_size,
        verbose=args.verbose,
        output_file=args.output,
        output_format=args.format,
        model_folder=args.model_folder,
        cards_folder=args.cards_folder,
    )


if __name__ == "__main__":
    main()

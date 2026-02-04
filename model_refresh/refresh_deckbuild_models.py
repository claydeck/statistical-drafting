#!/usr/bin/env python3
"""
Deckbuild Model Refresh Automation Script

This script automates the process of checking for and downloading new MTG game data
from 17lands, then retraining deckbuild models when updates are detected.

Unlike draft models which use draft_data, deckbuild models use game_data which contains
deck and sideboard information from actual games played.
"""

import json
import os
import sys
import requests
import shutil
from datetime import datetime
from typing import Dict, Optional, Tuple

# Add the parent directory to sys.path to import statisticaldeckbuild
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statisticaldeckbuild as sdb

# Import get_latest_set from the same directory
from get_latest_set import get_latest_set_info, get_file_last_modified

# Configuration
DECKBUILD_TRACKER_PATH = "deckbuild_tracker.json"
GAME_DATA_PATH = "../data/17lands/"


def load_deckbuild_tracker() -> Dict:
    """Load the deckbuild data tracker JSON file."""
    if os.path.exists(DECKBUILD_TRACKER_PATH):
        with open(DECKBUILD_TRACKER_PATH, 'r') as f:
            return json.load(f)
    else:
        return {
            "most_recent_set": None,
            "game_data_updates": {},
            "last_check_timestamp": None,
            "last_training_logs": [],
            "notes": "This file tracks game_data updates for deckbuild model training."
        }


def save_deckbuild_tracker(data: Dict) -> None:
    """Save the deckbuild data tracker JSON file."""
    data["last_check_timestamp"] = datetime.now().isoformat()
    with open(DECKBUILD_TRACKER_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated deckbuild tracker: {DECKBUILD_TRACKER_PATH}")


def download_file(url: str, destination: str) -> bool:
    """Download a file from URL to destination path."""
    try:
        print(f"Downloading {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        os.makedirs(os.path.dirname(destination), exist_ok=True)

        with open(destination, 'wb') as f:
            shutil.copyfileobj(response.raw, f)

        print(f"Downloaded to {destination}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def get_game_data_url(set_code: str, draft_mode: str) -> str:
    """Construct the game data URL for a given set and mode."""
    return f"https://17lands-public.s3.amazonaws.com/analysis_data/game_data/game_data_public.{set_code}.{draft_mode}Draft.csv.gz"


def check_game_data_available(set_code: str, draft_mode: str) -> Optional[str]:
    """
    Check if game data is available for a given set and mode.

    Returns:
        Last modified date string if available, None otherwise.
    """
    url = get_game_data_url(set_code, draft_mode)
    try:
        response = requests.head(url, timeout=10)
        if response.status_code == 200:
            last_modified = response.headers.get('Last-Modified')
            if last_modified:
                dt = datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S %Z')
                return dt.strftime('%Y-%m-%d')
        return None
    except Exception:
        return None


def check_and_download_game_data(
    tracker_data: Dict,
    set_code: str,
    draft_mode: str = "Premier"
) -> Tuple[bool, Optional[str]]:
    """
    Check if game data needs updating and download if necessary.

    Args:
        tracker_data: Current tracker data
        set_code: Set abbreviation (e.g., "FDN")
        draft_mode: Draft mode ("Premier", "Trad", etc.)

    Returns:
        Tuple of (data_updated, last_modified_date)
    """
    print(f"\nChecking game data for {set_code} {draft_mode}Draft...")

    # Get current tracking info
    game_data_updates = tracker_data.get("game_data_updates", {})
    tracker_key = f"{set_code}_{draft_mode}"
    current_date = game_data_updates.get(tracker_key)

    # Check if game data is available
    url = get_game_data_url(set_code, draft_mode)
    last_modified = check_game_data_available(set_code, draft_mode)

    if not last_modified:
        print(f"Game data not available for {set_code} {draft_mode}Draft")
        return False, None

    # Check if update is needed
    if current_date == last_modified:
        print(f"Game data up to date ({last_modified})")
        return False, last_modified

    if current_date is None:
        print(f"First download - game data available: {last_modified}")
    else:
        print(f"Game data updated: {current_date} -> {last_modified}")

    # Download game data
    gz_path = os.path.join(GAME_DATA_PATH, f"game_data_public.{set_code}.{draft_mode}Draft.csv.gz")
    if download_file(url, gz_path):
        # Update tracker
        game_data_updates[tracker_key] = last_modified
        tracker_data["game_data_updates"] = game_data_updates
        print(f"Game data download tracked: {tracker_key} ({last_modified})")
        return True, last_modified
    else:
        print(f"Failed to download game data for {set_code} {draft_mode}Draft")
        return False, None


def run_deckbuild_training(set_code: str, draft_mode: str) -> Tuple[bool, Dict]:
    """Run the deckbuild training pipeline for a given set and draft mode."""
    try:
        print(f"\nStarting deckbuild training for {set_code} {draft_mode}Draft...")

        # Change to notebooks directory (expected context for relative paths)
        original_cwd = os.getcwd()
        notebooks_dir = os.path.join(os.path.dirname(os.getcwd()), "notebooks")

        if os.path.exists(notebooks_dir):
            os.chdir(notebooks_dir)
            print(f"Changed working directory to: {os.getcwd()}")
        else:
            print(f"Warning: notebooks directory not found, using current directory")

        try:
            print(f"Calling sdb.default_deckbuild_pipeline...")
            print(f"   Parameters: set={set_code}, mode={draft_mode}")
            sys.stdout.flush()

            training_info = sdb.default_deckbuild_pipeline(
                set_abbreviation=set_code,
                draft_mode=draft_mode,
                overwrite_dataset=True,
            )

            print(f"Training Summary:")
            print(f"   Experiment: {training_info['experiment_name']}")
            print(f"   Training date: {training_info['training_date']}")
            print(f"   Training examples: {training_info['training_examples']:,}")
            print(f"   Validation examples: {training_info['validation_examples']:,}")
            print(f"   Best validation accuracy: {training_info['validation_accuracy']:.2f}%")
            print(f"   Best epoch: {training_info['num_epochs']}")

            print(f"Deckbuild training completed for {set_code} {draft_mode}Draft")
            return True, training_info
        finally:
            os.chdir(original_cwd)

    except Exception as e:
        print(f"Deckbuild training failed for {set_code} {draft_mode}Draft: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def refresh_deckbuild_models(
    set_code: Optional[str] = None,
    draft_mode: str = "Premier",
    force: bool = False,
) -> Dict:
    """
    Main function to refresh deckbuild models.

    Args:
        set_code: Set abbreviation. If None, uses the latest set from 17lands.
        draft_mode: Draft mode ("Premier", "Trad", etc.)
        force: If True, force retraining even if data hasn't changed.

    Returns:
        Summary dictionary with results.
    """
    print("Starting Deckbuild Model Refresh")
    print("=" * 50)

    # Load tracker
    tracker_data = load_deckbuild_tracker()
    print(f"Loaded tracker data")

    # Get set code
    if set_code is None:
        print("\nFetching latest set information from 17lands...")
        latest_set_info = get_latest_set_info()

        if not latest_set_info.get("success"):
            print(f"Failed to get latest set info: {latest_set_info}")
            return {"success": False, "error": "Failed to get latest set info"}

        set_code = latest_set_info.get("most_recent_set")
        print(f"Latest set: {set_code}")

    tracker_data["most_recent_set"] = set_code

    # Check and download game data
    data_updated, last_modified = check_and_download_game_data(
        tracker_data, set_code, draft_mode
    )

    # Run training if data was updated or force flag is set
    training_logs = []

    if data_updated or force:
        if force and not data_updated:
            print("\nForce flag set - retraining even though data hasn't changed")

        success, training_info = run_deckbuild_training(set_code, draft_mode)
        if success:
            training_logs.append(training_info)
        else:
            print(f"Training failed for {set_code} {draft_mode}Draft")
    else:
        print("\nNo updates detected - skipping training")

    # Update tracker
    if training_logs:
        tracker_data["last_training_logs"] = training_logs
        tracker_data["last_training_timestamp"] = datetime.now().isoformat()

    save_deckbuild_tracker(tracker_data)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Set: {set_code}")
    print(f"Draft Mode: {draft_mode}")
    print(f"Game Data Downloaded: {'Yes' if data_updated else 'No'}")
    print(f"Game Data Last Modified: {last_modified or 'N/A'}")

    if training_logs:
        print(f"\nTraining Results:")
        for log in training_logs:
            print(f"   {log['experiment_name']}: {log['validation_accuracy']:.2f}% accuracy")
            print(f"     {log['training_examples']:,} training examples, {log['num_epochs']} epochs")
    elif not data_updated and not force:
        print("\nAll data is up to date - no action needed!")

    return {
        "success": True,
        "set_code": set_code,
        "draft_mode": draft_mode,
        "data_updated": data_updated,
        "last_modified": last_modified,
        "training_logs": training_logs,
    }


def main():
    """Main entry point with command line argument support."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Refresh deckbuild models from 17lands game data"
    )
    parser.add_argument(
        "--set",
        type=str,
        default=None,
        help="Set abbreviation (e.g., FDN). If not specified, uses latest set."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="Premier",
        choices=["Premier", "Trad", "PickTwo"],
        help="Draft mode (default: Premier)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if data hasn't changed"
    )

    args = parser.parse_args()

    refresh_deckbuild_models(
        set_code=args.set,
        draft_mode=args.mode,
        force=args.force,
    )


if __name__ == "__main__":
    main()

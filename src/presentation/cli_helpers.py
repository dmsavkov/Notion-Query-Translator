"""CLI helpers for interactive prompts and complex terminal UI logic.
"""

from typing import List, Dict, Optional
import questionary

async def cli_disambiguator(title: str, options: List[Dict[str, str]]) -> Optional[str]:
    """Interactive questionary prompt for resource resolution."""
    # Build list of display choices
    choices = []
    for opt in options:
        name = opt['title']
        url = opt.get('url', '')
        short_url = url.split('/')[-1] if url else "link"
        choices.append(
            questionary.Choice(
                title=f"{name} ([link]{short_url}[/link])",
                value=opt['id']
            )
        )
    
    choices.append(questionary.Choice(title="[None of above - Cancel]", value=None))

    answer = await questionary.select(
        f"Multiple pages found for '{title}'. Please select the correct one:",
        choices=choices,
        style=questionary.Style([
            ('qmark', 'fg:#673ab7 bold'),       # token in front of the question
            ('question', 'bold'),               # question text
            ('answer', 'fg:#f44336 bold'),      # submitted answer text behind the question
            ('pointer', 'fg:#673ab7 bold'),     # pointer used in select and checkbox prompts
            ('highlighted', 'fg:#673ab7 bold'), # pointed-at choice in select and checkbox prompts
            ('selected', 'fg:#cc9900'),         # checkbox indicator
            ('separator', 'fg:#cc5454'),        # separator in lists
            ('instruction', ''),                # user instructions for select, raw_select, checkbox
            ('text', ''),                       # plain text
            ('disabled', 'fg:#858585 italic')   # disabled choices for select and checkbox prompts
        ])
    ).ask_async()

    return answer

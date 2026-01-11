"""
Command-line interface for lifecycle-planning.

Provides commands for running demonstrations, examples, and analysis.

Usage:
    lifecycle --version
    lifecycle demo
    lifecycle demo isabela
    lifecycle info
"""

import sys
import click
from pathlib import Path

from lifecycle_planning import __version__


@click.group()
@click.version_option(version=__version__, prog_name="lifecycle-planning")
def cli():
    """
    Lifecycle Financial Planning toolkit.

    A Python library for lifecycle financial planning based on
    "Lifetime Financial Advice" by Ibbotson, Milevsky, Chen, and Zhu.
    """
    pass


@cli.command()
@click.argument('profile', required=False, default=None)
@click.option('--list', 'list_demos', is_flag=True, help='List available demos')
def demo(profile, list_demos):
    """
    Run demonstration examples.

    PROFILE: Name of the demo to run (quick, isabela). Defaults to quick demo.

    Examples:
        lifecycle demo           # Run quick demo
        lifecycle demo isabela   # Run Isabela case study
        lifecycle demo --list    # List all available demos
    """
    if list_demos:
        click.echo("Available demos:")
        click.echo("  quick     - Quick demonstration of library features")
        click.echo("  isabela   - Comprehensive Isabela case study")
        return

    if profile is None or profile.lower() == 'quick':
        # Run quick demo
        click.echo("Running quick demo...\n")
        try:
            # Add examples directory to path (for development/repo usage)
            import lifecycle_planning
            pkg_path = Path(lifecycle_planning.__file__).parent.parent.parent
            examples_path = pkg_path / 'examples'
            if examples_path.exists():
                sys.path.insert(0, str(pkg_path))

            from examples.quick_demo import demo as run_quick_demo
            run_quick_demo()
        except (ImportError, ModuleNotFoundError):
            # If running from installed package, guide user
            click.echo("Quick demo is available in the examples/ directory")
            click.echo("Clone the repository to run examples:")
            click.echo("  git clone https://github.com/Druhayes/lifecycle-planning.git")
            sys.exit(1)

    elif profile.lower() == 'isabela':
        # Run Isabela example
        click.echo("Running Isabela case study...\n")
        try:
            # Add examples directory to path (for development/repo usage)
            import lifecycle_planning
            pkg_path = Path(lifecycle_planning.__file__).parent.parent.parent
            examples_path = pkg_path / 'examples'
            if examples_path.exists():
                sys.path.insert(0, str(pkg_path))

            from examples.isabela import main as run_isabela
            run_isabela()
        except (ImportError, ModuleNotFoundError):
            # If running from installed package, guide user
            click.echo("Isabela example is available in the examples/ directory")
            click.echo("Clone the repository to run examples:")
            click.echo("  git clone https://github.com/Druhayes/lifecycle-planning.git")
            sys.exit(1)

    else:
        click.echo(f"Unknown demo: {profile}")
        click.echo("Available demos: quick, isabela")
        click.echo("Use 'lifecycle demo --list' to see all demos")
        sys.exit(1)


@cli.command()
def info():
    """
    Display package information.

    Shows version, installation location, and other package metadata.
    """
    import lifecycle_planning

    click.echo("=" * 70)
    click.echo("LIFECYCLE FINANCIAL PLANNING")
    click.echo("=" * 70)
    click.echo(f"Version: {__version__}")
    click.echo(f"Location: {Path(lifecycle_planning.__file__).parent}")
    click.echo()
    click.echo("Description:")
    click.echo("  A comprehensive Python library for lifecycle financial planning")
    click.echo("  based on 'Lifetime Financial Advice: Human Capital, Asset")
    click.echo("  Allocation, and Insurance' by Ibbotson, Milevsky, Chen, and Zhu.")
    click.echo()
    click.echo("Features:")
    click.echo("  - Gompertz mortality modeling")
    click.echo("  - CRRA utility and preference functions")
    click.echo("  - Human capital and liability calculations")
    click.echo("  - Mean-variance portfolio optimization")
    click.echo("  - Optimal spending rules")
    click.echo("  - Annuity pricing and analysis")
    click.echo("  - Individual economic balance sheets")
    click.echo()
    click.echo("Documentation: https://lifecycle-planning.readthedocs.io")
    click.echo("Repository: https://github.com/Druhayes/lifecycle-planning")
    click.echo("=" * 70)


if __name__ == '__main__':
    cli()

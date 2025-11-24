"""Docker-style credential helper pattern for secure credential storage.

This module provides a modular credential storage system that auto-detects
the best available secure storage backend for the current platform.
"""

import json
import base64
import platform
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict
import os


class CredentialStore(ABC):
    """Abstract base class for credential storage backends."""

    @abstractmethod
    def get(self, account: str) -> Optional[Dict]:
        """Retrieve credentials for an account.

        Args:
            account: Email account identifier

        Returns:
            Dictionary containing credentials, or None if not found
        """
        pass

    @abstractmethod
    def store(self, account: str, credentials: Dict) -> None:
        """Store credentials for an account.

        Args:
            account: Email account identifier
            credentials: Dictionary containing credentials to store
        """
        pass

    @abstractmethod
    def erase(self, account: str) -> None:
        """Remove credentials for an account.

        Args:
            account: Email account identifier
        """
        pass

    @abstractmethod
    def list(self) -> list:
        """List all stored account identifiers.

        Returns:
            List of account email addresses
        """
        pass


class KeychainStore(CredentialStore):
    """macOS Keychain Access credential storage."""

    SERVICE_NAME = "inbox-reaper-oauth"

    def get(self, account: str) -> Optional[Dict]:
        try:
            result = subprocess.run([
                'security', 'find-generic-password',
                '-s', self.SERVICE_NAME,
                '-a', account,
                '-w'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                decoded = base64.b64decode(result.stdout.strip())
                return json.loads(decoded)
        except Exception:
            pass
        return None

    def store(self, account: str, credentials: Dict) -> None:
        encoded = base64.b64encode(json.dumps(credentials).encode()).decode()

        # Delete existing entry first
        self.erase(account)

        subprocess.run([
            'security', 'add-generic-password',
            '-s', self.SERVICE_NAME,
            '-a', account,
            '-w', encoded,
            '-U'  # Update if exists
        ], check=True)

    def erase(self, account: str) -> None:
        subprocess.run([
            'security', 'delete-generic-password',
            '-s', self.SERVICE_NAME,
            '-a', account
        ], capture_output=True)  # Ignore errors if doesn't exist

    def list(self) -> list:
        try:
            result = subprocess.run([
                'security', 'dump-keychain'
            ], capture_output=True, text=True)
            # Parse output for our service entries
            accounts = []
            for line in result.stdout.split('\n'):
                if f'"{self.SERVICE_NAME}"' in line and '"acct"' in line:
                    # Extract account from line
                    account = line.split('"acct"')[1].split('"')[1]
                    accounts.append(account)
            return accounts
        except Exception:
            return []


class DPAPIStore(CredentialStore):
    """Windows DPAPI via PowerShell credential storage."""

    def get(self, account: str) -> Optional[Dict]:
        try:
            ps_script = f'''
            $cred = Get-StoredCredential -Target "inbox-reaper-oauth:{account}" -Type Generic
            if ($cred) {{
                [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
                    [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($cred.Password)
                )
            }}
            '''
            result = subprocess.run([
                'powershell', '-Command', ps_script
            ], capture_output=True, text=True)

            if result.returncode == 0 and result.stdout.strip():
                decoded = base64.b64decode(result.stdout.strip())
                return json.loads(decoded)
        except Exception:
            pass
        return None

    def store(self, account: str, credentials: Dict) -> None:
        encoded = base64.b64encode(json.dumps(credentials).encode()).decode()

        ps_script = f'''
        $pass = ConvertTo-SecureString -String "{encoded}" -AsPlainText -Force
        $cred = New-Object System.Management.Automation.PSCredential("{account}", $pass)
        Write-StoredCredential -Target "inbox-reaper-oauth:{account}" -Type Generic -Credential $cred
        '''

        subprocess.run([
            'powershell', '-Command', ps_script
        ], check=True)

    def erase(self, account: str) -> None:
        ps_script = f'Remove-StoredCredential -Target "inbox-reaper-oauth:{account}" -Type Generic'
        subprocess.run(['powershell', '-Command', ps_script], capture_output=True)

    def list(self) -> list:
        ps_script = '''
        Get-StoredCredential -AsCredentialObject |
        Where {$_.Target -like "inbox-reaper-oauth:*"} |
        ForEach {$_.Target.Replace("inbox-reaper-oauth:", "")}
        '''
        result = subprocess.run([
            'powershell', '-Command', ps_script
        ], capture_output=True, text=True)

        if result.returncode == 0:
            return [acc.strip() for acc in result.stdout.split('\n') if acc.strip()]
        return []


class SecretServiceStore(CredentialStore):
    """Linux Secret Service (GNOME Keyring, KWallet, etc.) credential storage."""

    def __init__(self):
        import secretstorage
        self.connection = secretstorage.dbus_init()
        self.collection = secretstorage.get_default_collection(self.connection)

    def get(self, account: str) -> Optional[Dict]:
        items = self.collection.search_items({
            'application': 'inbox-reaper-oauth',
            'account': account
        })

        for item in items:
            secret = item.get_secret().decode('utf-8')
            return json.loads(base64.b64decode(secret))
        return None

    def store(self, account: str, credentials: Dict) -> None:
        encoded = base64.b64encode(json.dumps(credentials).encode()).decode()

        # Remove existing
        self.erase(account)

        self.collection.create_item(
            f'OAuth credentials for {account}',
            {'application': 'inbox-reaper-oauth', 'account': account},
            encoded.encode('utf-8'),
            replace=True
        )

    def erase(self, account: str) -> None:
        items = self.collection.search_items({
            'application': 'inbox-reaper-oauth',
            'account': account
        })
        for item in items:
            item.delete()

    def list(self) -> list:
        items = self.collection.search_items({
            'application': 'inbox-reaper-oauth'
        })
        return [item.get_attributes()['account'] for item in items]


class FileStore(CredentialStore):
    """Fallback: Encrypted file storage using OS user encryption."""

    def __init__(self):
        if platform.system() == 'Windows':
            base = os.environ.get('APPDATA', '.')
        elif platform.system() == 'Darwin':
            base = os.path.expanduser('~/Library/Application Support')
        else:
            base = os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))

        self.store_dir = Path(base) / 'inbox-reaper-oauth'
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Make directory readable only by owner
        if platform.system() != 'Windows':
            self.store_dir.chmod(0o700)

    def _get_file(self, account: str) -> Path:
        """Get file path for an account, sanitizing the account name."""
        # Sanitize account name for filename
        safe_name = "".join(c if c.isalnum() or c in ('-', '_', '.', '@') else '_' for c in account)
        return self.store_dir / f"{safe_name}.json"

    def get(self, account: str) -> Optional[Dict]:
        file = self._get_file(account)
        if file.exists():
            with open(file, 'r') as f:
                return json.load(f)
        return None

    def store(self, account: str, credentials: Dict) -> None:
        file = self._get_file(account)
        with open(file, 'w') as f:
            json.dump(credentials, f)

        # Make file readable only by owner
        if platform.system() != 'Windows':
            file.chmod(0o600)

    def erase(self, account: str) -> None:
        file = self._get_file(account)
        if file.exists():
            file.unlink()

    def list(self) -> list:
        accounts = []
        for file in self.store_dir.glob('*.json'):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if 'email' in data:
                        accounts.append(data['email'])
            except Exception:
                pass
        return accounts


def get_credential_store() -> CredentialStore:
    """Auto-detect and return the best available credential store.

    Returns:
        CredentialStore instance for the current platform

    The function tries to use platform-native secure storage:
    - macOS: Keychain
    - Windows: DPAPI via PowerShell
    - Linux: Secret Service (GNOME Keyring, KWallet, etc.)
    - Fallback: Encrypted file storage
    """
    system = platform.system()

    if system == 'Darwin':
        return KeychainStore()

    elif system == 'Windows':
        # Check if PowerShell credential manager is available
        try:
            result = subprocess.run([
                'powershell', '-Command',
                'Get-Command Get-StoredCredential -ErrorAction SilentlyContinue'
            ], capture_output=True)
            if result.returncode == 0:
                return DPAPIStore()
        except Exception:
            pass

    elif system == 'Linux':
        try:
            import secretstorage
            return SecretServiceStore()
        except ImportError:
            pass

    # Fallback to file store
    print("Warning: Using file-based credential storage. Consider installing:")
    if system == 'Linux':
        print("  - pip install secretstorage")
    elif system == 'Windows':
        print("  - Install-Module -Name CredentialManager")

    return FileStore()

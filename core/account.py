"""One person's account on one market: their broker and their own state
(day, peak, pause, pending orders, dopamine...). The brain's decisions are the
same for everyone; sizes, risk and money are per account."""
from .context import mkt
from .storage import Storage


class Account:
    def __init__(self, user, st: Storage):
        self.user = user
        self.name = user.name
        self.st = st
        self.creds = mkt().account_creds(user) or {}
        self.broker = mkt().make_broker(st, self)

    def get(self, key, default=None):
        return self.st.get(f"{self.name}:{key}", default)

    def set(self, key, value):
        self.st.set(f"{self.name}:{key}", value)

    def __repr__(self):
        return f"Account({self.name}@{mkt().name})"

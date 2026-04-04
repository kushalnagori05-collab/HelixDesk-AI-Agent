"""KnowledgeBase — in-memory KB for auto-reply lookups and self-update."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KBEntry:
    """A single knowledge base article."""
    entry_id: str
    category: str
    question_keywords: list[str]
    answer_text: str
    source: str  # "seed" | "auto_learned"
    use_count: int = 0


class KnowledgeBase:
    """Simple in-memory knowledge base supporting category lookup and auto-learning.

    Seeded with 24 entries (3 per category × 8 categories). Supports similarity
    lookup by category matching and optional self-update from resolved queries.
    """

    def __init__(self):
        self._auto_id_counter: int = 0
        self._entries: list[KBEntry] = self._seed_entries()

    def lookup(self, category: str, sentiment: float) -> tuple[KBEntry | None, float]:
        """Find best matching KB entry for this email category.

        Args:
            category: The email's category string.
            sentiment: The email's sentiment intensity (unused for matching,
                reserved for future weighting).

        Returns:
            (entry, similarity_score) where:
              - similarity = 1.0 if exact category match found
              - similarity = 0.5 if partial match (shared keyword overlap)
              - similarity = 0.0 and entry=None if no match
        """
        exact_matches = [e for e in self._entries if e.category == category]
        if exact_matches:
            # Pick the entry with the highest use_count (most validated)
            best = max(exact_matches, key=lambda e: e.use_count)
            best.use_count += 1
            return best, 1.0

        # Partial match: look for entries whose keywords overlap with the category name
        category_words = set(category.replace("_", " ").lower().split())
        best_entry: KBEntry | None = None
        best_overlap = 0

        for entry in self._entries:
            kw_words = set()
            for kw in entry.question_keywords:
                kw_words.update(kw.lower().split())
            overlap = len(category_words & kw_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_entry = entry

        if best_entry is not None and best_overlap > 0:
            best_entry.use_count += 1
            return best_entry, 0.5

        return None, 0.0

    def add_entry(self, category: str, question_keywords: list[str], answer_text: str) -> None:
        """Auto-learn a new KB entry from a resolved query.

        Args:
            category: The category this entry covers.
            question_keywords: Keywords associated with the question.
            answer_text: The answer/resolution text.
        """
        self._auto_id_counter += 1
        self._entries.append(KBEntry(
            entry_id=f"auto_{self._auto_id_counter}",
            category=category,
            question_keywords=question_keywords,
            answer_text=answer_text,
            source="auto_learned",
            use_count=0,
        ))

    def _seed_entries(self) -> list[KBEntry]:
        """Return 24 seed entries: 3 per category × 8 categories.

        Uses deterministic IDs (seed_00 .. seed_23) for reproducibility.
        """
        seed_data = [
            # --- login_failure ---
            ("login_failure",
             ["can't login", "password", "access denied"],
             "Please reset your password using the 'Forgot Password' link on the login page. If the issue persists, clear your browser cache and cookies, then try again."),
            ("login_failure",
             ["two-factor", "2FA", "authentication code"],
             "If your 2FA code isn't arriving, check your spam folder. You can also use backup codes from your security settings, or contact support to temporarily disable 2FA."),
            ("login_failure",
             ["locked out", "account locked", "too many attempts"],
             "Your account is temporarily locked after multiple failed login attempts. Please wait 30 minutes and try again, or use the 'Unlock Account' link sent to your registered email."),
            # --- billing_dispute ---
            ("billing_dispute",
             ["double charge", "billed twice", "duplicate"],
             "We apologize for the duplicate charge. Our billing team will review and process a refund for the extra charge within 5-7 business days."),
            ("billing_dispute",
             ["unrecognized charge", "unknown fee", "unexpected billing"],
             "The charge may be from a subscription renewal or add-on service. Please check your subscription dashboard for active services. If the charge is unauthorized, we'll initiate a refund."),
            ("billing_dispute",
             ["processing fee", "hidden fee", "extra charge"],
             "Our pricing page lists all applicable fees. Processing fees apply to certain payment methods. You can switch to direct bank transfer to avoid these fees."),
            # --- refund_request ---
            ("refund_request",
             ["refund", "money back", "return"],
             "Refund requests are processed within 7-10 business days after the return is confirmed. You can track the refund status in your account under 'Orders > Refunds'."),
            ("refund_request",
             ["cancel subscription", "cancelled but charged"],
             "If you were charged after cancellation, this may be due to a billing cycle overlap. We'll verify the cancellation date and process any applicable refund."),
            ("refund_request",
             ["trial period", "free trial charge"],
             "If you cancelled within the free trial window and were still charged, please provide your cancellation confirmation. We'll process a full refund immediately."),
            # --- product_defect ---
            ("product_defect",
             ["bug", "crash", "error", "not working"],
             "We're aware of this issue and our engineering team is working on a fix. As a workaround, please try clearing your cache or using an incognito browser window."),
            ("product_defect",
             ["update broke", "after update", "broken feature"],
             "We apologize for the disruption caused by the recent update. A hotfix is being deployed. In the meantime, you can roll back to the previous version in Settings > Version History."),
            ("product_defect",
             ["data loss", "corrupted", "sync issue"],
             "We take data integrity very seriously. Please check your backup settings under Account > Data Management. Our team will investigate the sync issue and help recover any lost data."),
            # --- shipping_delay ---
            ("shipping_delay",
             ["late delivery", "delayed", "not arrived"],
             "We apologize for the delay. Please share your order number and we'll contact the carrier for an updated delivery estimate. If the package is lost, we'll arrange a replacement."),
            ("shipping_delay",
             ["tracking", "no updates", "stuck in transit"],
             "Tracking updates can sometimes lag by 24-48 hours. If there's no update after 48 hours, please contact us with your order number and we'll escalate with the carrier."),
            ("shipping_delay",
             ["express shipping", "paid for fast", "priority delivery"],
             "We apologize that your express shipment didn't arrive on time. We'll refund the shipping fee difference and investigate the delay with our logistics partner."),
            # --- account_locked ---
            ("account_locked",
             ["unlock account", "account suspended", "restricted"],
             "Your account may have been locked due to unusual activity. Please verify your identity by clicking the verification link sent to your registered email to unlock your account."),
            ("account_locked",
             ["security alert", "suspicious activity", "flagged"],
             "We detected unusual login patterns and temporarily secured your account. Please complete the security verification process to restore full access."),
            ("account_locked",
             ["deactivated", "team member", "reactivate"],
             "Team member accounts can be reactivated by an admin user. Go to Admin > Team Management > Inactive Users and click 'Reactivate' next to the user's name."),
            # --- data_privacy ---
            ("data_privacy",
             ["data download", "export my data", "personal data"],
             "You can download your data from Account Settings > Privacy > 'Download My Data'. The export will be ready within 24 hours and you'll receive an email with the download link."),
            ("data_privacy",
             ["data retention", "how long", "keep data"],
             "We retain user data for 2 years after account deletion as required by financial regulations. You can review our full data retention policy at our Privacy Center."),
            ("data_privacy",
             ["opt out", "third party", "data sharing", "delete account"],
             "You can opt out of third-party data sharing in Settings > Privacy > Data Sharing Preferences. To delete your account entirely, submit a request through our Privacy Center."),
            # --- general_query ---
            ("general_query",
             ["business hours", "phone support", "contact"],
             "Our phone support is available Monday-Friday, 9 AM to 6 PM IST. You can also reach us 24/7 via email at support@helixdesk.com or through our live chat."),
            ("general_query",
             ["student discount", "educational", "pricing"],
             "Yes, we offer a 50% discount for verified students and educational institutions. Apply through our Education page with a valid .edu email address."),
            ("general_query",
             ["upgrade plan", "change plan", "API documentation"],
             "You can upgrade your plan at any time from Account > Subscription. The price difference will be prorated. API documentation is available at docs.helixdesk.com/api."),
        ]

        return [
            KBEntry(
                entry_id=f"seed_{i:02d}",
                category=cat,
                question_keywords=kws,
                answer_text=ans,
                source="seed",
            )
            for i, (cat, kws, ans) in enumerate(seed_data)
        ]

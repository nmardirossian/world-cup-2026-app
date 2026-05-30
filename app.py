"""World Cup betting pool app.

Uses Streamlit + Google Sheets (via st-gsheets-connection) as the backend.
Sheet tabs: Users (Username, PIN),
            Matches (Match_ID, Team_A, Team_B, Actual_Result),
            Predictions (Username, Match_ID, Predicted_Result).
Results and predictions are scorelines like "2-1".
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection

USERS_TAB = "Users"
MATCHES_TAB = "Matches"
PREDICTIONS_TAB = "Predictions"

# Global edit lock: all predictions freeze at this moment.
PACIFIC = ZoneInfo("America/Los_Angeles")
EDIT_LOCK_AT = datetime(2026, 6, 11, 12, 0, tzinfo=PACIFIC)

# Short TTL so writes are visible quickly without hammering the API.
READ_TTL_SECONDS = 30

SCORING_RULES_MD = """
**Scoring**
- Exact scoreline → **4 pts**
- Correct winner *and* goal difference → **2 pts**
- Correct winner only → **1 pt**
- Otherwise → 0 pts
"""

BUY_IN_USD = 5
PRIZE_PERCENTAGES = (0.70, 0.20, 0.10)  # 1st, 2nd, 3rd


# ---------- Data access ----------------------------------------------------

def get_conn() -> GSheetsConnection:
    return st.connection("gsheets", type=GSheetsConnection)


def _to_text(value: object) -> str:
    """Sheet-cell → clean string. Drops trailing .0 from integer-valued floats."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def load_users() -> pd.DataFrame:
    df = get_conn().read(worksheet=USERS_TAB, ttl=READ_TTL_SECONDS)
    df = df.dropna(how="all")
    df["PIN"] = df["PIN"].apply(_to_text)
    df["Username"] = df["Username"].astype(str).str.strip()
    return df[df["Username"] != ""].reset_index(drop=True)


def _match_sort_key(match_id: str) -> tuple[int, str]:
    """Natural sort: M1, M2, ..., M10 (not M1, M10, M2)."""
    digits = "".join(ch for ch in str(match_id) if ch.isdigit())
    return (int(digits) if digits else 10**9, str(match_id))


def load_matches(ttl: int = READ_TTL_SECONDS) -> pd.DataFrame:
    df = get_conn().read(worksheet=MATCHES_TAB, ttl=ttl)
    df = df.dropna(how="all")
    df["Match_ID"] = df["Match_ID"].apply(_to_text)
    df = df[df["Match_ID"] != ""].copy()
    if "Actual_Result" in df.columns:
        df["Actual_Result"] = df["Actual_Result"].fillna("").astype(str).str.strip()
    else:
        df["Actual_Result"] = ""
    df["_sort"] = df["Match_ID"].map(_match_sort_key)
    return df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def load_predictions(ttl: int = READ_TTL_SECONDS) -> pd.DataFrame:
    df = get_conn().read(worksheet=PREDICTIONS_TAB, ttl=ttl)
    df = df.dropna(how="all")
    if df.empty:
        return pd.DataFrame(columns=["Username", "Match_ID", "Predicted_Result"])
    df["Username"] = df["Username"].astype(str).str.strip()
    df["Match_ID"] = df["Match_ID"].apply(_to_text)
    df["Predicted_Result"] = df["Predicted_Result"].astype(str).str.strip()
    return df[(df["Username"] != "") & (df["Match_ID"] != "")].reset_index(drop=True)


# ---------- Scoring --------------------------------------------------------

def parse_score(s: object) -> tuple[int, int] | None:
    """Parse 'A-B' into (a, b). Returns None for blank / malformed values."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    text = str(s).strip()
    if not text or "-" not in text:
        return None
    a, _, b = text.partition("-")
    try:
        return int(a.strip()), int(b.strip())
    except ValueError:
        return None


def score_points(predicted: tuple[int, int] | None, actual: tuple[int, int] | None) -> int:
    if predicted is None or actual is None:
        return 0
    pa, pb = predicted
    aa, ab = actual
    if pa == aa and pb == ab:
        return 4
    pred_winner = (pa > pb) - (pa < pb)
    actual_winner = (aa > ab) - (aa < ab)
    if pred_winner != actual_winner:
        return 0
    if (pa - pb) == (aa - ab):
        return 2
    return 1


def compute_leaderboard(
    users: pd.DataFrame, matches: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    decided = matches.copy()
    decided["_actual"] = decided["Actual_Result"].apply(parse_score)
    decided = decided[decided["_actual"].notna()][["Match_ID", "_actual"]]

    if decided.empty or predictions.empty:
        totals = pd.DataFrame({"Username": users["Username"], "Total_Points": 0})
    else:
        joined = predictions.merge(decided, on="Match_ID", how="inner")
        joined["_predicted"] = joined["Predicted_Result"].apply(parse_score)
        joined["points"] = [
            score_points(p, a) for p, a in zip(joined["_predicted"], joined["_actual"])
        ]
        totals = joined.groupby("Username", as_index=False)["points"].sum()
        totals = users[["Username"]].merge(totals, on="Username", how="left")
        totals["points"] = totals["points"].fillna(0).astype(int)
        totals = totals.rename(columns={"points": "Total_Points"})

    totals = totals.sort_values(
        ["Total_Points", "Username"], ascending=[False, True]
    ).reset_index(drop=True)
    # Competition ranking: tied scores share the lower rank (1, 1, 3, ...).
    totals["Rank"] = (
        totals["Total_Points"].rank(method="min", ascending=False).astype(int)
    )
    totals["Winnings"] = compute_winnings(totals).apply(lambda x: f"${x:.2f}")
    return totals[["Rank", "Username", "Total_Points", "Winnings"]]


def compute_winnings(board: pd.DataFrame) -> pd.Series:
    """Tie-aware payout per user. Tied groups pool the prizes for the positions
    they collectively occupy, then split evenly. Positions past 3rd pay nothing."""
    n_users = len(board)
    pot = n_users * BUY_IN_USD
    winnings = pd.Series(0.0, index=board.index)

    pos = 0
    while pos < n_users:
        score = board.iloc[pos]["Total_Points"]
        n_tied = 1
        while pos + n_tied < n_users and board.iloc[pos + n_tied]["Total_Points"] == score:
            n_tied += 1

        covered = PRIZE_PERCENTAGES[pos : pos + n_tied]
        share = sum(covered) / n_tied if n_tied else 0
        for i in range(pos, pos + n_tied):
            winnings.iloc[i] = pot * share

        pos += n_tied

    return winnings


# ---------- Writes ---------------------------------------------------------

def save_predictions_bulk(username: str, updates: list[tuple[str, str]]) -> None:
    """Upsert many predictions in one sheet write. updates = [(match_id, "A-B"), ...].

    Reads with ttl=0 so back-to-back saves see each other's writes (otherwise
    a stale cached read would overwrite the previous save). Does NOT clear the
    app-wide cache afterwards — that would change the bracket's data prop
    mid-edit and cause data_editor to drop unsaved keystrokes."""
    if not updates:
        return
    conn = get_conn()
    df = load_predictions(ttl=0)
    for match_id, predicted_result in updates:
        mask = (df["Username"] == username) & (df["Match_ID"] == match_id)
        if mask.any():
            df.loc[mask, "Predicted_Result"] = predicted_result
        else:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "Username": username,
                                "Match_ID": match_id,
                                "Predicted_Result": predicted_result,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    conn.update(worksheet=PREDICTIONS_TAB, data=df)


def save_match_results(updates: dict[str, str]) -> None:
    """Update Actual_Result for the given match IDs. Rewrites the Matches sheet.

    Same pattern as save_predictions_bulk: read fresh, don't clear the app-wide
    cache afterwards (that can drop unsaved data_editor edits)."""
    if not updates:
        return
    conn = get_conn()
    df = load_matches(ttl=0)
    for match_id, result in updates.items():
        df.loc[df["Match_ID"] == match_id, "Actual_Result"] = result
    # Preserve the original column order, not the natural-sort order.
    conn.update(
        worksheet=MATCHES_TAB,
        data=df[["Match_ID", "Team_A", "Team_B", "Actual_Result"]],
    )


def update_user_pin(username: str, new_pin: str) -> None:
    conn = get_conn()
    df = load_users()
    df.loc[df["Username"] == username, "PIN"] = new_pin
    conn.update(worksheet=USERS_TAB, data=df[["Username", "PIN"]])
    st.cache_data.clear()


# ---------- Misc helpers ---------------------------------------------------

def format_countdown(delta: timedelta) -> str:
    total_minutes = int(delta.total_seconds() // 60)
    if total_minutes <= 0:
        return "now"
    days, rem = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def table_height(n_rows: int) -> int:
    """Pixels needed to show all rows of a Streamlit dataframe/data_editor
    without an internal scrollbar. Streamlit's canvas grid uses ~35px per row
    regardless of our app-wide CSS font bump."""
    return 38 + 35 * max(n_rows, 1)


def get_admins() -> set[str]:
    try:
        admins = st.secrets["app"]["admins"]
    except (FileNotFoundError, KeyError, AttributeError, TypeError):
        return set()
    return {str(a).strip() for a in admins}


def is_locked() -> bool:
    """True if past the global lock OR admin preview toggle is on."""
    if datetime.now(PACIFIC) >= EDIT_LOCK_AT:
        return True
    return bool(st.session_state.get("preview_locked", False))


# ---------- Auth -----------------------------------------------------------

def login_view() -> None:
    users = load_users()
    if users.empty:
        st.error("No users configured in the Users tab.")
        return

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        st.title("World Cup Pool")
        st.caption("Sign in")
        with st.form("login"):
            username = st.selectbox("Username", users["Username"].tolist())
            pin = st.text_input("PIN", type="password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            row = users[users["Username"] == username].iloc[0]
            if str(pin).strip() == row["PIN"]:
                st.session_state["auth_user"] = username
                # Pre-warm caches so the bracket renders instantly after rerun.
                with st.spinner("Loading your bracket..."):
                    load_matches()
                    load_predictions()
                st.rerun()
            else:
                st.error("Incorrect PIN.")


def logout() -> None:
    st.session_state.pop("auth_user", None)
    st.rerun()


# ---------- Views ----------------------------------------------------------

def bracket_view(username: str) -> None:
    st.subheader("My Bracket")
    matches = load_matches()
    preds = load_predictions()
    user_preds = preds[preds["Username"] == username].set_index("Match_ID")["Predicted_Result"]

    if matches.empty:
        st.info("No matches scheduled yet.")
        return

    now = datetime.now(PACIFIC)
    actually_locked = now >= EDIT_LOCK_AT
    preview = st.session_state.get("preview_locked", False)
    locked = actually_locked or preview

    lock_str = EDIT_LOCK_AT.strftime("%b %d, %Y %I:%M %p %Z")
    if actually_locked:
        st.warning(f"Predictions locked as of {lock_str}.")
    elif preview:
        st.info(f"🔍 Admin preview — bracket isn't actually locked yet (locks at {lock_str}).")
    else:
        countdown = format_countdown(EDIT_LOCK_AT - now)
        st.info(
            f"Predictions lock in **{countdown}** (at {lock_str}). "
            "Enter scores as `A-B` (e.g. `2-1`), then click **Save**."
        )
        submitted = sum(1 for mid in matches["Match_ID"] if parse_score(user_preds.get(mid, "")))
        st.caption(f"You have submitted **{submitted} of {len(matches)}** predictions.")

    rows = []
    for _, match in matches.iterrows():
        existing = parse_score(user_preds.get(match["Match_ID"], ""))
        display = f"{existing[0]}-{existing[1]}" if existing else ""
        rows.append(
            {
                "Match_ID": match["Match_ID"],
                "Team A": match["Team_A"],
                "Predicted Score": display,
                "Team B": match["Team_B"],
            }
        )
    # Rows are already in natural Match_ID order from load_matches.
    original = pd.DataFrame(rows).reset_index(drop=True)

    left, _ = st.columns([3, 1])
    with left:
        # Placeholder for the Save button. Filled in after we know how many
        # edits are pending, so the button label can show the count.
        save_slot = st.empty()
        edited = st.data_editor(
            original,
            column_config={
                "Match_ID": st.column_config.TextColumn("Match ID", disabled=True),
                "Team A": st.column_config.TextColumn("Team A", disabled=True),
                "Predicted Score": st.column_config.TextColumn(
                    "Predicted Score", help="Format: A-B (e.g. 2-1)."
                ),
                "Team B": st.column_config.TextColumn("Team B", disabled=True),
            },
            column_order=["Match_ID", "Team A", "Predicted Score", "Team B"],
            hide_index=True,
            disabled=locked,
            width="stretch",
            height=table_height(len(original)),
            key=f"bracket_editor_{username}",
        )

    if locked:
        return

    # Per-session memory of what's been saved. Lets the Save button correctly
    # disable itself after a save, even while the read cache is still stale.
    last_saved = st.session_state.setdefault(f"last_saved_{username}", {})

    updates: list[tuple[str, str]] = []
    errors: list[str] = []
    for i, row in edited.iterrows():
        match_id = str(row["Match_ID"])
        new_str = str(row["Predicted Score"] or "").strip()
        if not new_str:
            continue
        parsed = parse_score(new_str)
        if parsed is None:
            errors.append(f"{match_id}: '{new_str}' is not a valid score. Use A-B (e.g. 2-1).")
            continue
        normalized = f"{parsed[0]}-{parsed[1]}"
        if last_saved.get(match_id) == normalized:
            continue
        orig_str = str(original.iloc[i]["Predicted Score"] or "").strip()
        if normalized == orig_str:
            last_saved[match_id] = normalized
            continue
        updates.append((match_id, normalized))

    for err in errors:
        st.error(err)

    n = len(updates)
    button_label = f"💾 Save {n} change(s)" if n else "No changes to save"
    if save_slot.button(
        button_label,
        disabled=(n == 0),
        type="primary",
        key=f"save_predictions_{username}",
    ):
        save_predictions_bulk(username, updates)
        for match_id, pred in updates:
            last_saved[match_id] = pred
        st.toast(f"Saved {n} prediction(s) ✅")
        st.rerun()


def all_brackets_view(username: str) -> None:
    """Wide read-only matrix of everyone's predictions, available after lock."""
    st.subheader("All Brackets")
    matches = load_matches()
    predictions = load_predictions()

    if matches.empty:
        st.info("No matches scheduled yet.")
        return
    if predictions.empty:
        st.info("No predictions submitted.")
        return

    pivot = predictions.pivot_table(
        index="Match_ID", columns="Username", values="Predicted_Result", aggfunc="first"
    )
    base = matches[["Match_ID", "Team_A", "Team_B", "Actual_Result"]].copy()
    base["Matchup"] = base["Team_A"] + " vs " + base["Team_B"]

    df = base.merge(pivot, on="Match_ID", how="left")
    df["_sort"] = df["Match_ID"].map(_match_sort_key)
    df = df.sort_values("_sort").drop(columns=["_sort", "Team_A", "Team_B"])

    user_cols = sorted([c for c in pivot.columns if c != username])
    if username in pivot.columns:
        user_cols = [username] + user_cols
    df = df[["Match_ID", "Matchup", "Actual_Result"] + user_cols]
    df = df.rename(columns={"Actual_Result": "Actual"}).fillna("—").replace({"": "—"})

    st.caption("Your column is shown first.")
    left, _ = st.columns([3, 1])
    with left:
        st.dataframe(df, hide_index=True, width="stretch", height=table_height(len(df)))


def my_results_view(username: str) -> None:
    st.subheader("My Results")
    matches = load_matches()
    preds = load_predictions()
    user_preds = preds[preds["Username"] == username].set_index("Match_ID")["Predicted_Result"]

    if matches.empty:
        st.info("No matches scheduled yet.")
        return

    rows = []
    total = 0
    for _, m in matches.iterrows():
        match_id = m["Match_ID"]
        predicted_str = user_preds.get(match_id, "") or ""
        actual_str = m["Actual_Result"] or ""
        predicted = parse_score(predicted_str)
        actual = parse_score(actual_str)

        if actual is None:
            score_display = "—"
        else:
            pts = score_points(predicted, actual)
            total += pts
            score_display = str(pts)

        rows.append(
            {
                "Match ID": match_id,
                "Team A": m["Team_A"],
                "Predicted Result": f"{predicted[0]}-{predicted[1]}" if predicted else "—",
                "Team B": m["Team_B"],
                "Actual Result": f"{actual[0]}-{actual[1]}" if actual else "—",
                "Score": score_display,
            }
        )

    df = pd.DataFrame(rows)

    st.caption(f"Total points so far: **{total}**")
    left, _ = st.columns([3, 1])
    with left:
        st.dataframe(df, hide_index=True, width="stretch", height=table_height(len(df)))


def admin_view() -> None:
    st.subheader("Admin — enter match results")
    st.caption(
        "Format scores as `A-B` (e.g. `2-1`). Leave blank for matches not yet "
        "played, then click **Save**."
    )
    matches = load_matches()
    if matches.empty:
        st.info("No matches to manage.")
        return

    original = matches[["Match_ID", "Team_A", "Team_B", "Actual_Result"]].copy()

    left, _ = st.columns([3, 1])
    with left:
        save_slot = st.empty()
        edited = st.data_editor(
            original,
            column_config={
                "Match_ID": st.column_config.TextColumn("Match ID", disabled=True),
                "Team_A": st.column_config.TextColumn("Team A", disabled=True),
                "Team_B": st.column_config.TextColumn("Team B", disabled=True),
                "Actual_Result": st.column_config.TextColumn("Actual Result"),
            },
            hide_index=True,
            width="stretch",
            height=table_height(len(original)),
            key="admin_results_editor",
        )

    last_saved = st.session_state.setdefault("admin_last_saved_results", {})

    updates: dict[str, str] = {}
    errors: list[str] = []
    for i, row in edited.iterrows():
        match_id = str(row["Match_ID"])
        new = str(row["Actual_Result"] or "").strip()
        old = str(original.iloc[i]["Actual_Result"] or "").strip()
        parsed = parse_score(new) if new else None
        if new and parsed is None:
            errors.append(f"{match_id}: '{new}' is not a valid score (use A-B).")
            continue
        normalized = f"{parsed[0]}-{parsed[1]}" if parsed else ""
        if last_saved.get(match_id) == normalized:
            continue
        if normalized == old:
            last_saved[match_id] = normalized
            continue
        updates[match_id] = normalized

    for err in errors:
        st.error(err)

    n = len(updates)
    button_label = f"💾 Save {n} change(s)" if n else "No changes to save"
    if save_slot.button(
        button_label,
        disabled=(n == 0),
        type="primary",
        key="save_admin_results",
    ):
        save_match_results(updates)
        for match_id, value in updates.items():
            last_saved[match_id] = value
        st.toast(f"Saved {n} result(s) ✅")
        st.rerun()


def pin_change_form(username: str) -> None:
    with st.expander("Change PIN"):
        with st.form("change_pin", clear_on_submit=True):
            current = st.text_input("Current PIN", type="password")
            new = st.text_input("New PIN", type="password")
            confirm = st.text_input("Confirm new PIN", type="password")
            submitted = st.form_submit_button("Update")
        if submitted:
            users = load_users()
            row = users[users["Username"] == username].iloc[0]
            if current.strip() != row["PIN"]:
                st.error("Current PIN is incorrect.")
            elif not new.strip():
                st.error("New PIN can't be empty.")
            elif new != confirm:
                st.error("New PIN entries don't match.")
            else:
                update_user_pin(username, new.strip())
                st.success("PIN updated.")


def leaderboard_view() -> None:
    st.subheader("Leaderboard")
    board = compute_leaderboard(load_users(), load_matches(), load_predictions())

    pot = len(board) * BUY_IN_USD
    st.caption(
        f"\\${BUY_IN_USD} buy-in · pot **\\${pot}** · split 70/20/10 for 1st/2nd/3rd. "
        "Ties pool the prizes for the positions they share and split evenly."
    )

    def _highlight(row: pd.Series) -> list[str]:
        if row["Rank"] == 1:
            return ["background-color: #ffd700; font-weight: bold;"] * len(row)
        return [""] * len(row)

    styled = board.style.apply(_highlight, axis=1)
    left, _ = st.columns([1, 3])
    with left:
        st.dataframe(
            styled,
            hide_index=True,
            width="stretch",
            height=table_height(len(board)),
        )


# ---------- Entry point ----------------------------------------------------

GLOBAL_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"], .stApp { font-size: 18px; }
.stMarkdown p, .stMarkdown li, label, .stTextInput input, .stSelectbox div,
.stNumberInput input, .stButton button, .stForm, .stCaption,
[data-testid="stDataFrame"] div, [data-testid="stTable"] td,
[data-testid="stTable"] th { font-size: 1.05rem; }
h1 { font-size: 2.4rem; }
h2 { font-size: 1.9rem; }
h3 { font-size: 1.55rem; }
</style>
"""


def main() -> None:
    st.set_page_config(page_title="World Cup Pool", page_icon="⚽", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    if "auth_user" not in st.session_state:
        login_view()
        return

    user = st.session_state["auth_user"]
    is_admin = user in get_admins()

    with st.sidebar:
        st.write(f"Signed in as **{user}**")
        col_refresh, col_logout = st.columns(2)
        if col_refresh.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
        if col_logout.button("Log out"):
            logout()
        st.divider()
        st.markdown(SCORING_RULES_MD)
        st.divider()
        pin_change_form(user)
        if is_admin:
            st.divider()
            st.toggle(
                "Preview locked view",
                key="preview_locked",
                help="Render the bracket as if predictions were already locked. Admin-only.",
            )

    locked = is_locked()
    tab_names = ["My Bracket"]
    if locked:
        tab_names.append("All Brackets")
    tab_names += ["Leaderboard", "My Results"]
    if is_admin:
        tab_names.append("Admin")
    tabs = dict(zip(tab_names, st.tabs(tab_names)))

    with tabs["My Bracket"]:
        bracket_view(user)
    if "All Brackets" in tabs:
        with tabs["All Brackets"]:
            all_brackets_view(user)
    with tabs["Leaderboard"]:
        leaderboard_view()
    with tabs["My Results"]:
        my_results_view(user)
    if "Admin" in tabs:
        with tabs["Admin"]:
            admin_view()


if __name__ == "__main__":
    main()

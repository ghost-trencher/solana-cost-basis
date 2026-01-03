import streamlit as st
import pandas as pd
import requests
import networkx as nx
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO
from collections import deque
from datetime import datetime
import time

from solana.rpc.api import Client
from solders.pubkey import Pubkey as PublicKey

# Constants
RPC_URL = "https://api.mainnet-beta.solana.com"
client = Client(RPC_URL)

COINGECKO_HISTORY_URL = "https://api.coingecko.com/api/v3/coins/{id}/history"
COINGECKO_LIST_URL = "https://api.coingecko.com/api/v3/coins/list?include_platform=true"

SOL_MINT = "So11111111111111111111111111111111111111112"

@st.cache_data(ttl=86400)
def get_coingecko_mapping():
    try:
        response = requests.get(COINGECKO_LIST_URL)
        if response.status_code != 200:
            return {SOL_MINT: "solana"}
        coins = response.json()
        mapping = {SOL_MINT: "solana"}
        for coin in coins:
            platforms = coin.get("platforms", {})
            solana_mint = platforms.get("solana")
            if solana_mint:
                mapping[solana_mint] = coin["id"]
        return mapping
    except:
        return {SOL_MINT: "solana"}

COINGECKO_MAP = get_coingecko_mapping()

def get_asset_id(mint: str) -> str:
    return COINGECKO_MAP.get(mint, "solana")

def get_historical_price(asset_id: str, date_str: str) -> float:
    try:
        url = COINGECKO_HISTORY_URL.format(id=asset_id)
        params = {"date": date_str}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return 0.0
        data = response.json().get("market_data", {}).get("current_price", {}).get("usd", 0.0)
        return data or 0.0
    except:
        return 0.0

def fetch_transactions(address: str):
    try:
        pubkey = PublicKey.from_string(address)
    except:
        st.error("Invalid wallet address")
        return pd.DataFrame()

    all_transfers = []
    before = None
    limit = 1000

    progress = st.progress(0)
    status = st.empty()

    fetched_sigs = 0

    while True:
        status.text(f"Fetching signatures... ({len(all_transfers)} transfers found)")
        resp = client.get_signatures_for_address(pubkey, limit=limit, before=before)
        sigs = resp.value
        if not sigs:
            break

        for sig_info in sigs:
            tx_resp = client.get_transaction(sig_info.signature, encoding="jsonParsed", max_supported_transaction_version=0)
            tx = tx_resp.value
            if not tx:
                continue

            # Structure: tx.transaction.meta and tx.transaction.transaction.message.instructions
            encoded_tx = tx.transaction
            meta = encoded_tx.meta
            inner_transaction = encoded_tx.transaction

            if meta and meta.err:
                continue

            timestamp = datetime.utcfromtimestamp(sig_info.block_time) if sig_info.block_time else datetime.now()

            transfers = []
            message = inner_transaction.message
            for instr in message.instructions:
                # Some instructions are parsed (dict with 'parsed'), others partially decoded (accounts + data)
                if hasattr(instr, "parsed") and instr.parsed:
                    parsed = instr.parsed
                    if parsed.get("type") in ["transfer", "transferChecked"]:
                        info = parsed["info"]
                        amount_str = info.get("lamports") or info.get("tokenAmount", {}).get("uiAmountString", "0")
                        amount = float(amount_str or 0)
                        mint = info.get("mint", SOL_MINT)
                        source = info.get("source")
                        destination = info.get("destination")
                        if amount > 0:
                            transfers.append({
                                "asset": mint,
                                "amount": amount,
                                "from": source,
                                "to": destination,
                                "type": "Transfer",
                                "timestamp": timestamp,
                                "wallet": address,
                            })

            all_transfers.extend(transfers)

        fetched_sigs += len(sigs)
        progress.progress(min(fetched_sigs / (fetched_sigs + limit), 1.0))

        if len(sigs) < limit:
            break
        before = sigs[-1].signature
        time.sleep(0.3)

    df = pd.DataFrame(all_transfers)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    status.empty()
    progress.empty()
    return df

def classify_transfers(df: pd.DataFrame):
    df["category"] = "Transfer"
    return df

def discover_wallets(df: pd.DataFrame, main_wallet: str):
    G = nx.DiGraph()
    transfer_df = df[df["category"] == "Transfer"]
    for _, row in transfer_df.iterrows():
        if row["from"] and row["to"]:
            G.add_edge(row["from"], row["to"])
    potential = [n for n in G.nodes if n != main_wallet and 1 < G.degree(n) < 20]
    return list(set(potential))

def calculate_fifo(df: pd.DataFrame):
    if df.empty:
        return 0.0, ["No transactions"]
    df = df.copy()
    df["wallet"] = df["wallet"].fillna("unknown")
    groups = df.groupby(["wallet", "asset"])
    total_gains = 0.0
    alerts = []

    for (wallet, asset), group in groups:
        group = group.sort_values("timestamp")
        acquisitions = deque()
        for _, row in group.iterrows():
            asset_id = get_asset_id(row["asset"])
            date_str = row["timestamp"].strftime("%d-%m-%Y")
            price = get_historical_price(asset_id, date_str)
            amount = row["amount"]

            if row["to"] == wallet:  # Incoming
                basis = amount * price
                acquisitions.append((amount, basis))
            elif row["from"] == wallet:  # Outgoing
                remaining = amount
                cost_basis = 0.0
                while remaining > 0 and acquisitions:
                    aq_amt, aq_basis = acquisitions.popleft()
                    use = min(remaining, aq_amt)
                    cost_basis += (use / aq_amt) * aq_basis if aq_amt > 0 else 0
                    remaining -= use
                    if use < aq_amt:
                        remaining_basis = aq_basis * (aq_amt - use) / aq_amt
                        acquisitions.appendleft((aq_amt - use, remaining_basis))
                if remaining > 0:
                    alerts.append(f"Missing basis for {remaining:.4f} {asset[:8]}... in {wallet[:8]}...")
                proceeds = amount * price
                total_gains += proceeds - cost_basis

    return total_gains, alerts

def generate_pdf(gains: float, alerts: list, df: pd.DataFrame):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Solana Cost Basis Report (2025 IRS Compliant)", styles["Title"]))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"<b>Total Gains/Losses:</b> ${gains:,.2f}", styles["Heading1"]))
    elements.append(Spacer(1, 20))

    if alerts:
        elements.append(Paragraph("<b>Missing Basis Alerts:</b>", styles["Heading2"]))
        for a in alerts[:10]:
            elements.append(Paragraph(f"• {a}", styles["Normal"]))
        elements.append(Spacer(1, 12))

    data = [["Date", "Asset", "Amount", "From → To"]]
    for _, row in df.head(20).iterrows():
        data.append([
            row["timestamp"].strftime("%Y-%m-%d"),
            row["asset"][:8] + "...",
            f"{row['amount']:.4f}",
            f"{row['from'][:6] if row['from'] else ''} → {row['to'][:6] if row['to'] else ''}"
        ])
    table = Table(data, colWidths=[100, 100, 80, 200])
    table.setStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)])
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Mobile-Friendly UI
st.set_page_config(layout="centered", page_title="Solana Basis")
st.title("🪙 Free Solana Cost Basis Calculator")
st.caption("Mobile • 2025 IRS Per-Wallet FIFO • No signup")

with st.form("main_form"):
    main_wallet = st.text_input("Main Solana Wallet Address", placeholder="e.g. 4fYNw3dojNGzMwefMZ...")
    submitted = st.form_submit_button("Calculate Basis", use_container_width=True)

if submitted and main_wallet:
    df = fetch_transactions(main_wallet)
    if df.empty:
        st.warning("No transactions found — try a different wallet.")
    else:
        df = classify_transfers(df)

        with st.expander("🔍 Potential Owned Wallets — Check to Include"):
            potentials = discover_wallets(df, main_wallet)
            selected = []
            for p in potentials:
                count = df[df['from']==p].shape[0] + df[df['to']==p].shape[0]
                if st.checkbox(f"{p[:8]}...{p[-6:]} ({count} transfers)"):
                    selected.append(p)
                    extra = fetch_transactions(p)
                    if not extra.empty:
                        extra["wallet"] = p
                        df = pd.concat([df, extra], ignore_index=True)

        gains, alerts = calculate_fifo(df)
        st.success(f"**Total Capital Gains/Losses: ${gains:,.2f}**")

        pdf = generate_pdf(gains, alerts, df)
        st.download_button(
            "📄 Download PDF Report",
            pdf,
            "solana_cost_basis_2025.pdf",
            "application/pdf",
            use_container_width=True
        )

        st.info("Per-wallet FIFO • Missing basis = $0 (alerted) • Not tax advice — consult a CPA.")

st.markdown("---")
st.caption("Open-source • Free forever • Built Jan 2026")

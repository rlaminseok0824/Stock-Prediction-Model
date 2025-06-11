from datetime import datetime, timedelta
from math import sqrt
import os

import pandas as pd
import torch
import torch.nn as nn

from transformers import  GPT2Config, GPT2Model, GPT2Tokenizer
from layers.Embed import PatchEmbedding
import transformers

from models import RevIN

transformers.logging.set_verbosity_error()

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, cfg):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

        self.gpt2_config.num_hidden_layers = configs.llm_layers
        self.gpt2_config.output_attentions = True
        self.gpt2_config.output_hidden_states = True
        self.gpt2_config.n_embd = self.d_llm 
        self.gpt2_config.n_head = 8 
        self.gpt2_config.gradient_checkpointing = True

        try:
            self.llm_model = GPT2Model.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=True,
                config=self.gpt2_config,
            )
        except EnvironmentError:  # downloads model from HF is not already done
            print("Local model files not found. Attempting to download...")
            self.llm_model = GPT2Model.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=False,
                config=self.gpt2_config,
            )

        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=True
            )
        except EnvironmentError:  # downloads the tokenizer from HF if not already done
            print("Local tokenizer files not found. Atempting to download them..")
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=False
            )

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.description = 'Samsung Electronics stock serves as a major benchmark in global semiconductor markets and plays a central role in assessing investor sentiment toward South Korea’s tech industry.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        
        
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.news_df = pd.read_csv('data/stock/news.csv', parse_dates=['datetime'])
        self.news_df = self.news_df.set_index('datetime').sort_index()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
    
    def _get_news_for_date(self, target_date):
        """
        Retrieves the latest news entry on or before the target_date from the loaded DataFrame.
        Returns empty strings for summary and keywords if no relevant news is found.
        """
        if self.news_df.empty:
            return {"summary": "", "keywords": ""} # Return empty if no data loaded

        relevant_news = self.news_df.loc[self.news_df.index <= target_date]

        if not relevant_news.empty:
            latest_news = relevant_news.iloc[-1]
            return {"summary": latest_news['summary'], "keywords": latest_news['keywords']}
        
        return {"summary": "", "keywords": ""} #

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        company_profile = """
        Description:
        Samsung Electronics is a global technology leader headquartered in South Korea, known for its wide range of products including smartphones, semiconductors, home appliances, and display panels. As a key player in the global electronics and semiconductor industries, Samsung’s performance has significant influence on the tech and manufacturing sectors.
        
        Positive Factors:
        Strong semiconductor demand, New product launches (e.g. foldables, AI devices),Global tech innovation leadership,R&D investment and patent portfolio,Favorable FX (weak KRW), AI and data center growth, Strategic partnerships and M&A

        Negative Factors:
        Global inflation and high CPI, Interest rate hikes, Supply chain disruptions, Weak consumer demand (smartphones, appliances), Geopolitical tensions (e.g. China-US trade), Memory chip price volatility, Competitor pressure (e.g. TSMC, Apple)
        """
       
        prompt = []            
        for b in range(x_enc.shape[0]):
            first_forecast_stamp = x_mark_dec[0, 0].cpu().numpy().astype(int)

            forecast_year = first_forecast_stamp[0]
            forecast_month = first_forecast_stamp[1]
            forecast_day = first_forecast_stamp[2]
            prediction_date_for_news_search = datetime(forecast_year, forecast_month, forecast_day)

            two_weeks_ago_date = prediction_date_for_news_search - timedelta(weeks=2)
            one_week_ago_date = prediction_date_for_news_search - timedelta(weeks=1)

            two_weeks_ago_news = self._get_news_for_date(two_weeks_ago_date)
            one_week_ago_news = self._get_news_for_date(one_week_ago_date)

            recent_news_parts = []

            if two_weeks_ago_news['summary'] or two_weeks_ago_news['keywords']:
                recent_news_parts.append(f"""
    2 Weeks Ago
    Summary:
    {two_weeks_ago_news['summary']}
    Keyword:
    {two_weeks_ago_news['keywords']}
    """)
            
            if one_week_ago_news['summary'] or one_week_ago_news['keywords']:
                recent_news_parts.append(f"""
    1 Week Ago

    Summary:
    {one_week_ago_news['summary']}
    Keyword:
    {one_week_ago_news['keywords']}
    """)


            recent_news_str = "\n".join(recent_news_parts).strip() # Join parts with a newline, remove leading/trailing whitespace
            
            prompt_ = (
                f"<|start_prompt|>"
                f"Task description: Forecast the next {self.pred_len} stop stocks given the previous {self.seq_len} steps, "
                f"and considering company profile and recent news. "
                f"Company Profile:\n{company_profile.strip()}\n\n"
            )
            if recent_news_str: # Only add Recent News section if there's actual news content
                prompt_ += f"Recent News:\n{recent_news_str}\n"
            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        # GPT2 를 위해 1024로 max_length를 제한
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)
        
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)

        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        print("ReprogrammingLayer: d_model:", d_model, "n_heads:", n_heads, "d_keys:", d_keys, "d_llm:", d_llm)
        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding

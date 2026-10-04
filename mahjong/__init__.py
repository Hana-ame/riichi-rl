"""立直麻将 RL 环境包。

模块划分：
    tiles     牌的编码与类别判定
    hand      和了分解枚举、七对子、国士无双
    ruleset   规则开关与结算表
    yaku      役判定
    scoring   符/番汇总算分
    env       4 人混战对局引擎
    obs       观测编码与合法动作掩码
    policies  竞技场用的可插拔策略基线

对外入口：from mahjong import env, obs, policies
"""

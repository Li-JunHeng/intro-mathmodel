	1. 数据映射：速度，时间，车辆长度等
	2. 初始化：根据给定参数[自动驾驶汽车比例，车流个数，车道个数等]
	3. 对每个时间片执行以下几个步骤
		a. 模拟自动驾驶汽车的反应速度比普通的私家车还快，我们将一个时间片[也就是一个最小思考单元]分成三个部分
			i. 私家车根据当前的情况做出自己的决策
			ii. 自动驾驶汽车之间有自己独立的通信协同决策模式
		b. 换车道思考过程：
			i. 三车道换道模型
				1) 如果右车道安全，且比当前车道车况好[车距更大]，就换右车道
				2) 右车道路况没有比当前路况好，那么
					a) 如果左车道安全
					b) 当前车道不满足行驶条件
					c) 左车道路况比当前路况好
					d) 满足以上所有条件，换左车道
				3) 做出最后的换道决策，前面只是做出思考，有Pignore的概率拒绝换道[模拟人的因素]
				4) 这部分私家车和自动驾驶汽车暂时没有思考出区别
			ii. 考虑鸣笛效应
				1) 当n号车后面的车满足以下条件的时候，对n号车鸣笛
					a) n号车无法正常速度行驶
					b) n号车无法换道
				2) 当n号车被鸣笛之后：
					a) 如果右车道安全，且比当前车道车况好，就换右车道
					b) 如果右车道不能换，那么
						i) 如果左车道安全
						ii) 左车道满足当前的行驶条件[速度满足当前的速度]
					c) 决定换左车道
				3) 上面只是做出思考，没有做出最终决策，有Pignore的概率拒绝换道[模拟人的因素]
				4) 换道环节结束之后进入单车道思考过程
		c. 单车道思考过程
			i. 加速过程：Vn->min(vn+1,vmax)
				1) 私家车和自动驾驶汽车应该有不同的加速度，1应该取值不同，下同
			ii. 减速过程【应用加速效应模型进行改进】：vn->min(vn,,dn+v'n+1)[n号的加速度为1[最大加速度]，和考虑了前车的位移的情况下的两者的间距]
				1) 为了避免相撞
				2) v'n+1=vn+1 - min[v_max-1,v_(n+1),max(0,d_(n+1) - 1)]
				3) 对于自动驾驶汽车而言，安全距离减小，因为他能够灵活反应，对应在元胞机中，自动驾驶汽车是在私家车做出决策之后进行的决策，所以他所看到的私家车的vn是确定化的vn不会产生冲突
			iii. 随机慢化：
				1) 随机慢化的思想：模拟驾驶员状态导致的慢化
				2) 【应用巡航驾驶极限模型和VDR慢启动规则改进】
					a) 思想：
						i) 以期望的速度行驶[vmax]，不受随机慢化影响
						ii) 在上一刻静止的车辆，随机慢化的概率更大
					b) Pc = p(v) ->当v=0和v=max的时候，随机慢化概率远小于1，其他时间段随机慢化较高
					c) 进行随机慢化，在pc的概率下，vn->max(vn-1,0)
				3) 由于自动驾驶汽车的特性，其pc函数的值应该比私家车的低
			iv. 进行运动决策：xn->xn+vn
			v. 注意，这之前的步骤是每个步骤都要走一遍的，直到最后一步完成之后，再更新状态
	4. 一个时间片需要计算所有的车辆的状态之后并行同步更新，而不是计算一辆更新一辆

路段交替算法

1. 每次更新之后，把这个路段的前面max_v的数据交给上一个路段
2. 对于第一个路段，不需要这个操作
3. 最后一个路段，更新结束之后，没有下一个路段给他数据，他的数据恒为0

对于车辆位置的逻辑判断，需要把下一个路段的信息考虑进去

对于车辆位置的放置，只能放在本路段的位置
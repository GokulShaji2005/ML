
idx = np.argsort(x_test.flatten())


x_sorted = x_test.flatten()[idx]
y_test_sorted = y_test[idx]
y_pred_gd_sorted = y_pred_gd[idx]
y_pred_ne_sorted = y_pred_ne[idx]

# plt.figure(figsize=(12,5))
plt.figure(figsize=(8,5))
plt.plot(cost_history,color='navy',linewidth=2)

plt.title('Gradient descent cost convergence curve')
plt.grid(True,linestyle='--',alpha=0.6)
plt.show()


plt.subplot(1,2,2)
plt.scatter(x_test, y_test, color='blue', alpha=0.5, label='Actual Data')
plt.plot(x_sorted, y_pred_ne_sorted, color='green', linewidth=2,
         label='Normal Equation')
plt.xlabel("Average Rooms")
plt.ylabel("House Price")
plt.title("Normal Equation Regression")
plt.legend()

plt.tight_layout()
plt.show()


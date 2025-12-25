using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class DoggyAgent : Agent
{
    [Header("Target Settings")]
    public Transform target; // Куб-цель
    
    [Header("Agent Settings")]
    Rigidbody agentRb;
    EnvironmentParameters defaultParams;
    
    // Для отслеживания расстояния до цели
    float previousDistance;
    
    public override void Initialize()
    {
        agentRb = GetComponent<Rigidbody>();
        defaultParams = Academy.Instance.EnvironmentParameters;
        
        // Находим цель, если она не назначена вручную
        if (target == null)
        {
            GameObject targetObject = GameObject.FindGameObjectWithTag("Target");
            if (targetObject != null)
            {
                target = targetObject.transform;
            }
        }
    }
    
    public override void OnEpisodeBegin()
    {
        // Сброс позиции агента (опционально, если нужно)
        // transform.localPosition = new Vector3(0, 0.5f, 0);
        // agentRb.velocity = Vector3.zero;
        // agentRb.angularVelocity = Vector3.zero;
        
        // Инициализация предыдущего расстояния
        if (target != null)
        {
            previousDistance = Vector3.Distance(transform.localPosition, target.localPosition);
        }
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        // Наблюдение: позиция агента относительно цели
        if (target != null)
        {
            Vector3 localPosition = transform.localPosition;
            Vector3 targetLocalPosition = target.localPosition;
            
            // Направление к цели (3 значения)
            sensor.AddObservation(targetLocalPosition - localPosition);
            
            // Расстояние до цели (1 значение)
            float distanceToTarget = Vector3.Distance(localPosition, targetLocalPosition);
            sensor.AddObservation(distanceToTarget);
        }
        else
        {
            // Если цель не найдена, отправляем нули
            sensor.AddObservation(Vector3.zero);
            sensor.AddObservation(0f);
        }
        
        // Наблюдение: скорость агента (3 значения)
        sensor.AddObservation(agentRb.velocity);
        
        // Наблюдение: текущая позиция агента (3 значения)
        sensor.AddObservation(transform.localPosition);
        
        // Наблюдение: направление агента (3 значения - forward вектор)
        sensor.AddObservation(transform.forward);
    }
    
    public override void OnActionReceived(ActionBuffers actions)
    {
        // Получаем действия (обычно 2 действия: горизонтальное и вертикальное движение)
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];
        
        // Применяем движение к агенту
        Vector3 move = new Vector3(moveX, 0, moveZ);
        agentRb.AddForce(move * 5f, ForceMode.VelocityChange);
        
        // Вычисляем награды
        if (target != null)
        {
            float currentDistance = Vector3.Distance(transform.localPosition, target.localPosition);
            
            // 1. Награда за достижение цели (большая положительная награда)
            if (currentDistance < 1.5f) // Порог достижения цели
            {
                AddReward(10.0f);
                EndEpisode();
                return;
            }
            
            // 2. Формирующая награда: поощрение за приближение к цели
            float distanceDelta = previousDistance - currentDistance;
            AddReward(distanceDelta * 0.1f); // Масштабируем награду
            
            // 3. Небольшой штраф за каждый шаг (мотивирует к быстрому достижению цели)
            AddReward(-0.01f);
            
            // 4. Штраф за падение или выход за границы (если агент упал)
            if (transform.localPosition.y < -1f)
            {
                AddReward(-1.0f);
                EndEpisode();
                return;
            }
            
            previousDistance = currentDistance;
        }
        else
        {
            // Если цель не найдена, небольшая награда за исследование
            AddReward(-0.01f);
        }
    }
    
    // Функция для тестирования в режиме Heuristic Only
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        
        // Управление с клавиатуры для тестирования
        float moveX = 0f;
        float moveZ = 0f;
        
        if (Input.GetKey(KeyCode.W))
            moveZ = 1f;
        if (Input.GetKey(KeyCode.S))
            moveZ = -1f;
        if (Input.GetKey(KeyCode.A))
            moveX = -1f;
        if (Input.GetKey(KeyCode.D))
            moveX = 1f;
        
        continuousActionsOut[0] = moveX;
        continuousActionsOut[1] = moveZ;
    }
    
    // Опционально: визуализация для отладки
    void OnDrawGizmosSelected()
    {
        if (target != null)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawLine(transform.position, target.position);
        }
    }
}

